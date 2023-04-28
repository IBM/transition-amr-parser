# Standalone AMR parser from an existing trained APT model

import os
import time
import math
import copy
import signal
import argparse
from datetime import timedelta
import urllib.request
import zipfile

from ipdb import set_trace
import progressbar
from tqdm import tqdm
import torch
from fairseq import checkpoint_utils, utils
from fairseq.tokenizer import tokenize_line
from fairseq.models.bart import BARTModel

from fairseq_ext import options
from fairseq_ext.extract_bart.sentence_encoding import SentenceEncodingBART
from fairseq_ext.extract_bart.binarize_encodings import get_scatter_indices
from fairseq_ext.data.amr_action_pointer_dataset import collate
from transition_amr_parser.io import (
    read_tokenized_sentences, read_amr, read_config_variables
)
from fairseq_ext.utils import (
    post_process_action_pointer_prediction,
    post_process_action_pointer_prediction_bartsv,
    clean_pointer_arcs
)
from fairseq_ext.utils_import import import_user_module
from transition_amr_parser.amr import normalize, protected_tokenizer

from transition_amr_parser.make_sliding_splits import (
    adjust_sentence_ends,
    get_windows
)
from transition_amr_parser.force_overlap_actions import force_overlap
from transition_amr_parser.merge_sliding_splits import (
    merge_actions,
    check_pointers
)
from transition_amr_parser.amr_machine import play_all_actions
from transition_amr_parser.clbar import yellow_font
from transition_amr_parser.amr_machine import make_eos_force_actions


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Call parser from the command line'
    )
    parser.add_argument(
        '-i', '--in-tokenized-sentences',
        type=str,
        help='File with one __tokenized__ sentence per line'
    )
    parser.add_argument(
        '--in-doc',
        type=str,
        help='File with one __tokenized__ sentence per line and one newline separating each doc (and the end)'
    )
    parser.add_argument(
        '--in-amr',
        type=str,
        help='AMR in Penman format to align'
    )
    parser.add_argument(
        '--in-actions',
        type=str,
        help='action sequence (per token) for force decoding'
    )
    parser.add_argument(
        '--tokenize',
        action='store_true',
        help='Tokenize with a jamr-like tokenizer input sentences or AMR'
    )
    parser.add_argument(
        '--sliding',
        action='store_true',
        help='split into sliding windows (use --window-size and '
             '--window-overlap to adjust)'
    )
    parser.add_argument(
        "--window-size",
        help="size of sliding window",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--window-overlap",
        help="size of overlap between sliding windows",
        default=200,
        type=int,
    )
    parser.add_argument(
        '--beam',
        type=int,
        default=1,
        help='Beam decoding size'
    )
    parser.add_argument(
        '--nbest',
        type=int,
        default=1,
        help='Number of results per sentence'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        help='number of samples, if unset there is no sampling'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='temperature for sampling',
        default=1.0
    )
    parser.add_argument(
        '--sampling-topp',
        default=-1,
        type=float,
        help='sample from the smallest set whose cumulative probability mass '
             'exceeds p for next words (fairseq option)'
    )
    parser.add_argument(
        '--service',
        action='store_true',
        help='Prompt user for sentences'
    )
    parser.add_argument(
        '-c', '--in-checkpoint',
        type=str,
        help='one fairseq model checkpoint (or various, separated by :)'
    )
    parser.add_argument(
        '-m', '--model-name',
        type=str,
        help='name of model config (instead of checkpoint) and optionally'
             'seed separated by : e.g. amr2.0-structured-bart-large:42'
    )
    parser.add_argument(
        '-o', '--out-amr',
        type=str,
        help='File to store AMR in PENNMAN format'
    )
    parser.add_argument(
        '--out-actions',
        type=str,
        help='File containing parser actions'
    )
    parser.add_argument(
        '--out-tokens',
        type=str,
        help='File containing parser input tokens'
    )
    parser.add_argument(
        '--in-machine-config',
        type=str,
        help='Path to machine config file'
    )
    parser.add_argument(
        '--roberta-cache-path',
        type=str,
        help='Path to the pretrained BART LM'
    )
    parser.add_argument(
        '--roberta-batch-size',
        type=int,
        default=512,
        help='Batch size for roberta computation (watch for OOM)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for decoding (excluding roberta)'
    )
    # step by step parameters
    parser.add_argument(
        "--step-by-step",
        help="pause after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--set-trace",
        help="breakpoint after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--fp16",
        help="breakpoint after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--jamr",
        help="Add JAMR graph representation on meta-data",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--no-isi",
        help="Store ISI alignments in ::alignments field rather than node "
             "names. This helps with Smatch not supporting ISI",
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    # sanity checks
    assert (
        bool(args.in_tokenized_sentences) or bool(args.in_amr)
    ) or bool(args.service) or bool(args.in_doc), \
        "Must either specify --in-tokenized-sentences or --in-doc or set --service"

    if not (bool(args.model_name) ^ bool(args.in_checkpoint)):
        raise Exception("Use either --model-name or --in-checkpoint")

    if bool(args.in_actions) and bool(args.tokenize):
        raise Exception(
            "Remove --tokenize option when enabling --force-actions and"
            "provide tokenized input matching the dimensions of force actions."
        )
    if bool(args.in_actions) and bool(args.in_doc):
        print(yellow_font(
            'WARNING: Given force actions will be used superseding the force actions generated by the inhouse force action generator. Make sure it is in the correct format '))

    # num samples replaces beam search
    if args.num_samples:
        assert args.nbest == 1
        assert args.beam == 1

    if args.sampling_topp != -1:
        assert args.num_samples

    return args


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def load_models_and_task(args, use_cuda, task=None):
    # if `task` is not provided, it will be from the saved model args
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        args.path.split(':'),
        arg_overrides=args.model_overrides,
        task=task,
    )
    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            print('using fp16 for models')
            model.half()
        if use_cuda:
            print('using GPU for models')
            model.cuda()
        else:
            print('using CPU for models')

    # model = Model(models, task.target_dictionary)
    return models, model_args, task


def load_args_from_config(config_path):
    """Load args from bash configuration scripts"""
    # TODO there might be better ways; e.g. source the bash script in python
    # and use $BERT_LAYERS directly
    config_dict = {}
    with open(config_path, 'r') as f:
        for line in f:
            if line.strip():
                if not line.startswith('#') and '=' in line:
                    kv = line.strip().split('#')[0].strip().split('=')
                    assert len(kv) == 2
                    # remove the " in bach args
                    config_dict[kv[0]] = kv[1].replace('"', '')
    return config_dict


def load_roberta(name=None, roberta_cache_path=None, roberta_use_gpu=False):
    if not roberta_cache_path:
        # Load the Roberta Model from torch hub
        roberta = torch.hub.load('pytorch/fairseq', name)
    else:
        roberta = BARTModel.from_pretrained(
            roberta_cache_path,
            checkpoint_file='model.pt'
        )
    roberta.eval()
    if roberta_use_gpu:
        roberta.cuda()
    return roberta


def read_doc_text(in_file):
    sents = []
    docs = []
    for line in open(in_file, 'r').readlines():
        if line == '\n':
            docs.append(sents)
            sents = []
        else:
            sents.append(line.strip('\n'))
    if len(sents) > 0:
        docs.append(sents)

    return docs


def get_force_actions(docs, sen_ends=None):

    force_actions_set = []
    return_docs = []

    for idx, docsen in enumerate(docs):

        offset = 0
        if sen_ends is None:
            ends = []
            doc_toks = []
            for sen in docsen:

                ends.append(offset+len(sen)-1)
                offset += len(sen)
                doc_toks.extend(sen)

            force_actions = make_eos_force_actions(doc_toks, ends)
        else:
            force_actions = make_eos_force_actions(doc_toks, sen_ends[idx])
        force_actions_set.append(force_actions)
        return_docs.append(doc_toks)

    return force_actions_set, return_docs


def get_sliding_windows(tok_sentences, window_size, window_overlap,
                        force_actions=None):
    windowed_tok_sentences = []
    windowed_force_actions = []
    windowed_output_actions = []
    all_windows = []
    max_num_windows = 1

    for (i, sentence) in enumerate(tok_sentences):
        # if sentence end markers are availble, windows should respect that
        sentence_ends = []
        if force_actions:
            for (n, actions) in enumerate(force_actions[i]):
                if 'CLOSE_SENTENCE' in actions:
                    sentence_ends.append(n)
        if len(sentence_ends) == 0:
            sentence_ends = [len(sentence)-1]

        sentence_ends = adjust_sentence_ends(sentence_ends, window_size)

        # get windows respecting sentence ends
        windows = get_windows(sentence, window_size,
                              window_overlap, sentence_ends)
        all_windows.append(windows)
        if len(windows) > max_num_windows:
            max_num_windows = len(windows)

        windowed_tokenized_sent = []
        windowed_force_acts = []
        windowed_output_actions.append([])
        for (start, end) in windows:
            windowed_tokenized_sent.append(sentence[start:end+1])
            if force_actions:
                windowed_force_acts.append(force_actions[i][start:end+1])
            else:
                windowed_force_acts.append(None)
            windowed_output_actions[-1].append(None)

        windowed_tok_sentences.append(windowed_tokenized_sent)
        windowed_force_actions.append(windowed_force_acts)

    return (
        windowed_tok_sentences,
        windowed_force_actions,
        windowed_output_actions,
        all_windows,
        max_num_windows
    )


def get_sliding_output(tok_sentences, parser, gold_amrs=None,
                       force_actions=None, window_size=300, window_overlap=200,
                       batch_size=128, roberta_batch_size=128,
                       beam=1, jamr=False, no_isi=False):

    (
        windowed_tok_sentences,
        windowed_force_actions,
        windowed_output_actions,
        all_windows,
        max_num_windows
    ) = get_sliding_windows(
        tok_sentences=tok_sentences,
        window_size=window_size,
        window_overlap=window_overlap,
        force_actions=force_actions
    )

    for widx in range(max_num_windows):
        window_sentences = []
        window_actions = []
        wsidx2sidx = []
        for sidx in range(len(tok_sentences)):
            if len(windowed_tok_sentences[sidx]) > widx:
                wsidx2sidx.append(sidx)
                window_sentences.append(
                    windowed_tok_sentences[sidx][widx])
                window_actions.append(windowed_force_actions[sidx][widx])

        # Parse this (widx_th) window of all sentences
        num_window_sent = len(window_sentences)
        print(f'Parsing {num_window_sent} sentences in {widx} window')
        window_result = parser.parse_sentences(
            window_sentences,
            batch_size=batch_size,
            roberta_batch_size=roberta_batch_size,
            gold_amrs=gold_amrs,
            force_actions=window_actions,
            beam=beam,
            jamr=jamr,
            no_isi=no_isi
        )
        # get actions out of window_results
        # save actions in windowed_output_actions[sidx][widx]
        for residx, result in enumerate(window_result[1]):
            windowed_output_actions[wsidx2sidx[residx]
                                    ][widx] = result.action_history
        # change windowed_force_actions[:][widx+1]
        for sidx in range(len(tok_sentences)):
            if len(windowed_tok_sentences[sidx]) > widx+1:
                existing_f_actions = windowed_force_actions[sidx][widx+1]
                pred_f_actions = windowed_output_actions[sidx][widx]
                start_idx = all_windows[sidx][widx + 1][0] \
                    - all_windows[sidx][widx][0]
                new_f_actions = force_overlap(
                    pred_f_actions, existing_f_actions, start_idx)
                slen = len(windowed_tok_sentences[sidx][widx+1])
                if slen > len(new_f_actions):
                    new_f_actions = new_f_actions[:slen]
                else:
                    while len(new_f_actions) < slen:
                        new_f_actions.append([])
                windowed_force_actions[sidx][widx+1] = new_f_actions

    # merge all windowed_force_actions
    all_actions = []
    all_tokens = []
    for (didx, windows) in enumerate(all_windows):
        actions = []
        tokens = []
        for (i, window) in enumerate(windows):
            if i == 0:
                actions = windowed_output_actions[didx][i]
                tokens = windowed_tok_sentences[didx][i]
            else:
                new_actions = windowed_output_actions[didx][i]
                check_pointers(new_actions)
                merged_actions = merge_actions(
                    actions, new_actions, overlap_start=window[0])
                if check_pointers(merged_actions):
                    actions = merged_actions
                    window_tokens = windowed_tok_sentences[didx][i]
                    tokens.extend(window_tokens[len(tokens)-window[0]:])
                else:
                    print("did not complete merge for doc " + str(didx))
                    break
        all_actions.append(actions)
        all_tokens.append(tokens)

    # create final graphs
    annotations, machines = play_all_actions(
        all_tokens, all_actions, parser.machine_config)
    return annotations, machines


class AMRParser:
    def __init__(
        self,
        models,           # PyTorch model
        task,             # fairseq task
        src_dict,         # fairseq dict
        tgt_dict,         # fairseq dict
        machine_config,    # path to train.rules.json
        use_cuda,         #
        args,             # args for decoding
        model_args,       # args read from the saved model checkpoint
        to_amr=True,      # whether to output the final AMR graph
        embeddings=None,  # PyTorch RoBERTa model (if dealing with token input)
        inspector=None    # function to call after each step
    ):

        # member variables
        self.models = models
        self.task = task
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.embeddings = embeddings
        self.inspector = inspector
        self.machine_config = machine_config
        print("self.machine_config: ", self.machine_config)
        self.args = args
        self.model_args = model_args
        self.generator = self.task.build_generator(args, model_args)
        self.to_amr = to_amr

    @classmethod
    def default_args(cls, checkpoint=None, fp16=False):
        """Default args for generation"""
        default_args = ['dummy_data_folder',
                        '--emb-dir', 'dummy_emb_dir',
                        '--user-dir', './fairseq_ext',
                        # this is dummy; will be updated by the model args
                        '--task', 'amr_action_pointer_bart',
                        '--modify-arcact-score', '1',
                        '--beam', '1',
                        '--batch-size', '128',
                        '--remove-bpe',
                        '--path', checkpoint or 'dummy_model_path',
                        '--quiet',
                        '--src-fix-emb-use', '0',
                        '--results-path', 'dummy_results_path',
                        '--machine-rules', 'None']     # not currently used
        if fp16:
            default_args.append('--fp16')
        return default_args

    @classmethod
    def load(cls, model_name, seed=None, **kwargs):
        '''
        Initialize model from config file and seed
        '''
        # locate model config assuming it is contained in DATA inside repo
        # folder
        # TODO: Use environment variable to define storage place
        repo_folder = os.path.realpath(f'{__file__}/../../')
        config_path = f'{repo_folder}/configs/{model_name}.sh'
        config_path = os.path.realpath(config_path)
        if not os.path.isfile(config_path):
            raise Exception(f'{config_path} is not an existing model config')

        # locate model folder from config
        config_env_vars = read_config_variables(config_path)
        model_folder = config_env_vars['MODEL_FOLDER']
        model_folder = f'{repo_folder}/{model_folder}'
        if not os.path.isdir(model_folder):
            raise Exception(f'Missing model {model_folder}, is it available?')
        dec_checkpoint = config_env_vars['DECODING_CHECKPOINT']

        # get first checkpoint if no seed specified
        if seed is None:
            checkpoints = []
            for seedf in os.listdir(f'{model_folder}'):
                checkpoint = f'{model_folder}{seedf}/{dec_checkpoint}'
                if os.path.isfile(checkpoint):
                    checkpoints.append(checkpoint)
            if len(checkpoints) > 1:
                print(f'More than one seed picking {checkpoints[0]}')
            assert checkpoints, f'No completed checkpoints in {model_folder}'
            checkpoint = checkpoints[0]
        else:
            checkpoint = f'{model_folder}seed{seed}/{dec_checkpoint}'
            assert os.path.isfile(checkpoint), f"{checkpoint} missing"
        return cls.from_checkpoint(checkpoint, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name, dict_dir=None,
                        roberta_cache_path=None, fp16=False,
                        inspector=None, beam=1, nbest=1, num_samples=None,
                        sampling_topp=-1, temperature=1.0):
        """ Load model checkpoints from available model names; 
        Will check if the model is downloaded to cache, if not, download from cloud storage; 
        Below is a list of available modelnames to trun
            {
                'AMR3.0':'https://s3.us-east.cloud-object-storage.appdomain.cloud/cloud-object-storage-xc-cos-standard-htg/amr3.0-structured-bart-large-neur-al-sampling5.zip'
            }
        Args:
            modelname (str): a model name within our pretrained model library. 
            dict_dir (_type_, optional): _description_. Defaults to None.
            roberta_cache_path (_type_, optional): _description_. Defaults to None.
            fp16 (bool, optional): _description_. Defaults to False.
            inspector (_type_, optional): _description_. Defaults to None.
            beam (int, optional): _description_. Defaults to 1.
            nbest (int, optional): _description_. Defaults to 1.
            num_samples (_type_, optional): _description_. Defaults to None.
            sampling_topp (int, optional): _description_. Defaults to -1.
            temperature (float, optional): _description_. Defaults to 1.0.

        Raises:
            ValueError: The given model name doesn't match any model in library.
            Exception: _description_

        Returns:
            _type_: _description_
        """

        MODEL_NAMES = {
            'AMR3-structbart-L-smpl': 'amr3.0-structured-bart-large-neur-al-sampling5-seed42.zip',
            'AMR3-structbart-L': 'amr3.0-structured-bart-large-neur-al-seed42.zip',
            'AMR2-structbart-L': 'amr2.0-structured-bart-large-neur-al-seed42.zip',
            'AMR2-joint-ontowiki-seed42': 'amr2joint_ontowiki2_g2g-structured-bart-large-seed42.zip',
            'AMR2-joint-ontowiki-seed43': 'amr2joint_ontowiki2_g2g-structured-bart-large-seed43.zip',
            'AMR2-joint-ontowiki-seed44': 'amr2joint_ontowiki2_g2g-structured-bart-large-seed44.zip',
            'AMR3-joint-ontowiki-seed42': 'amr3joint_ontowiki2_g2g-structured-bart-large-seed42.zip',
            'AMR3-joint-ontowiki-seed43': 'amr3joint_ontowiki2_g2g-structured-bart-large-seed43.zip',
            'AMR3-joint-ontowiki-seed44': 'amr3joint_ontowiki2_g2g-structured-bart-large-seed44.zip',
            'doc-sen-conll-amr-seed42': 'both_doc+sen_trainsliding_ws400x100-seed42.zip'
        }

        if model_name not in MODEL_NAMES:
            raise ValueError('The model_name provided in not in our library; if you provide a path, \
                             use from_chackpoint() method instead')

        # get model class and model names
        zip_file = MODEL_NAMES[model_name]
        if "ontowiki2" in zip_file:
            model_class = zip_file.split("-structured-bart")[0]
        elif "amr2.0" in zip_file:
            model_class = "AMR2.0"
        else:
            model_class = "AMR3.0"

        if "doc" in model_name:
            eval_metric = "smatch"
        else:
            eval_metric = "wiki.smatch"

        model_name = zip_file.split("-seed")[0]
        seed = zip_file.split("-seed")[1][:2]

        # get cache storage dir, in which we save the downloaded models
        cache_dir = torch.hub._get_torch_home() + '/'
        cache_save_zip = cache_dir + zip_file

        # get checkpoint path
        model_dir_path = os.path.join(
            cache_dir, "DATA", model_class, "models", model_name, "seed"+str(seed))
        checkpoint_path = model_dir_path + "/checkpoint_"+eval_metric+"_top5-avg.pt"

        # get model url to download from
        model_url = "https://github.com/transition-amr-parsing/parser-model-hub/raw/main/models/" + zip_file

        if not os.path.isfile(checkpoint_path):
            print('Downloading model from library, and save to cache')
            # download and save to cache dir

            global pbar
            pbar = None

            def show_progress(block_num, block_size, total_size):
                global pbar
                if pbar is None:
                    pbar = progressbar.ProgressBar(maxval=total_size)
                    pbar.start()

                downloaded = block_num * block_size
                if downloaded < total_size:
                    pbar.update(downloaded)
                else:
                    pbar.finish()
                    pbar = None
            if not os.path.isfile(cache_save_zip):
                urllib.request.urlretrieve(
                    model_url, cache_save_zip, show_progress)
            else:
                print("a model zip file is already downloaded")

            if os.path.getsize(cache_save_zip) < 1000000:
                raise Exception(
                    "our model download limit is reached; please try downloading  again next week.")

            with zipfile.ZipFile(cache_dir + zip_file, 'r') as zip_ref:
                zip_ref.extractall(torch.hub._get_torch_home())
            print("downloaded model unzipped")
            assert os.path.isfile(
                checkpoint_path), 'checkpoint still not available after downloads;'
        else:
            print('model is already in cache')
        return cls.from_checkpoint(checkpoint_path, dict_dir,
                                   roberta_cache_path, fp16, inspector,
                                   beam, nbest, num_samples, sampling_topp,
                                   temperature)

    @classmethod
    def from_checkpoint(cls, checkpoint, dict_dir=None,
                        roberta_cache_path=None, fp16=False,
                        inspector=None, beam=1, nbest=1, num_samples=None,
                        sampling_topp=-1, temperature=1.0):
        '''
        Initialize model from checkpoint
        '''
        # load default args: some are dummy
        parser = options.get_interactive_generation_parser()
        # model path set here
        default_args = cls.default_args(checkpoint, fp16=fp16)
        args = options.parse_args_and_arch(parser, input_args=default_args)
        import_user_module(args)
        # when `input_args` is fed in, it overrides the command line input args
        # only one required positional argument for the argparser: data
        if args.max_tokens is None and args.max_sentences is None:
            args.max_tokens = 12000

        # load model (ensemble), further model args, and the fairseq task
        # ensure we read from the folder where this chekpoint is regardless of
        # what is saved on the checkpoint (we may have messed with the names)
        args.model_overrides = {
            'save_dir': os.path.dirname(checkpoint.split(':')[0])
        }
        if dict_dir is not None:
            args.model_overrides['data'] = dict_dir
            # otherwise, the default dict folder is read from the model args
        use_cuda = torch.cuda.is_available() and not args.cpu
        models, model_args, task = load_models_and_task(
            args, use_cuda, task=None
        )
        # overload some arguments
        # SequenceGenerator args: sampling_topk sampling_topp temperature
        args.beam = beam
        args.nbest = nbest
        args.sampling_topp = sampling_topp
        args.temperature = temperature
        if num_samples is not None:
            args.sampling = True

        # load pretrained Roberta model for source embeddings
        if model_args.pretrained_embed_dim == 768:
            pretrained_embed = 'bart.base'
        elif model_args.pretrained_embed_dim == 1024:
            pretrained_embed = 'bart.large'
        else:
            raise ValueError

        model_folder = os.path.dirname(checkpoint.split(':')[0])
        # config_data_path = None
        # for dfile in os.listdir(model_folder):
        #     if dfile.startswith('config.sh'):
        #         config_data_path = os.path.join(model_folder, dfile)
        #         break
        # assert config_data_path is not None, \
        #     'data configuration file not found'

        print("pretrained_embed: ", pretrained_embed)
        embeddings = SentenceEncodingBART(name=pretrained_embed)

        print("Finished loading models")

        # load other args
        machine_config = os.path.join(model_folder, 'machine_config.json')
        assert os.path.isfile(machine_config), f"Missing {machine_config}"

        return cls(models, task, task.src_dict, task.tgt_dict, machine_config,
                   use_cuda, args, model_args, to_amr=True,
                   embeddings=embeddings, inspector=inspector)

    def get_bert_features_batched(self, sentences, batch_size):
        bert_data = []
        num_batches = math.ceil(len(sentences)/batch_size)
        for i in tqdm(range(0, num_batches), desc='roberta'):
            batch = sentences[i * batch_size: i * batch_size + batch_size]
            batch_data = self.embeddings.extract_batch(batch)
            for i in range(0, len(batch)):
                bert_data.append((
                    copy.deepcopy(batch_data["word_features"][i]),
                    copy.deepcopy(batch_data["wordpieces_roberta"][i]),
                    copy.deepcopy(
                        batch_data["word2piece_scattered_indices"][i]
                    )
                ))
        print(len(bert_data))
        assert len(bert_data) == len(sentences)
        return bert_data

    def get_token_ids(self, sentence):
        return self.src_dict.encode_line(
            line=sentence,
            line_tokenizer=tokenize_line,
            add_if_not_exist=False,
            append_eos=False,
            reverse_order=False
        )

    def get_bart_features(self, sentences):
        bart_data = []
        for sent in sentences:
            wordpieces_roberta, word2piece = \
                self.embeddings.encode_sentence(sent)
            wordpieces_scattered_indices = get_scatter_indices(
                word2piece,
                reverse=True
            )
            bart_data.append((
                copy.deepcopy(wordpieces_roberta),
                copy.deepcopy(wordpieces_scattered_indices)
            ))
        print(len(bart_data))
        assert len(bart_data) == len(sentences)
        return bart_data

    def convert_sentences_to_data(self, sentences, batch_size,
                                  roberta_batch_size, gold_amrs=None,
                                  force_actions=None):

        assert gold_amrs is None or len(sentences) == len(gold_amrs)

        # extract RoBERTa features
        roberta_features = \
            self.get_bart_features(sentences)

        # organize data into a fairseq batch
        data = []
        for index, sentence in enumerate(sentences):
            ids = self.get_token_ids(sentence)
            wordpieces_roberta, word2piece_scattered_indices =\
                roberta_features[index]
            data.append({
                'id': index,
                'source': ids,
                'src_wordpieces': wordpieces_roberta,
                'src_wp2w': word2piece_scattered_indices,
                # original source tokens
                'src_tokens': tokenize_line(sentence),
                'gold_amr': None if gold_amrs is None else gold_amrs[index],
                'force_actions': None
                if force_actions is None else force_actions[index]
            })
        return data

    def get_iterator(self, samples, batch_size):
        batches = []
        for i in range(0, math.ceil(len(samples)/batch_size)):
            sample = samples[i * batch_size: i * batch_size + batch_size]
            batch = collate(
                sample, pad_idx=self.tgt_dict.pad(),
                eos_idx=self.tgt_dict.eos(),
                left_pad_source=True,
                left_pad_target=False,
                input_feeding=True,
                collate_tgt_states=False
            )

            # FIXME: This avoids adding collate code for each dataset but not
            # elegant
            if any(a.get('gold_amr', None) is not None for a in sample):
                # This also relies on ID bing the relative index as set above
                batch['gold_amr'] = [
                    samples[id]['gold_amr'] for id in batch['id']
                ]
            if any(a.get('force_actions', None) is not None for a in sample):
                # This also relies on ID bing the relative index as set above
                batch['force_actions'] = [
                    samples[id]['force_actions'] for id in batch['id']
                ]

            batches.append(batch)
        return batches

    def parse_batch(self, sample, to_amr=True):
        # TODO: pass arbitrary machines in generator to enable inspector and
        # other
        hypos = self.task.inference_step(
            self.generator, self.models, sample, self.args, prefix_tokens=None
        )

        predictions = []
        for i, sample_id in enumerate(sample['id'].tolist()):
            src_tokens = sample['src_sents'][i]
            target_tokens = None

            sample_predictions = []
            for j, hypo in enumerate(hypos[i][:self.args.nbest]):
                # args.nbest is default 1, i.e. saving only the top predictions
                if 'bartsv' in self.model_args.arch:
                    # shared vocabulary BART

                    actions_nopos, actions_pos, actions = \
                        post_process_action_pointer_prediction_bartsv(
                            hypo, self.tgt_dict
                        )

                    # FIXME: hypo['state_machine'].machine.action_history !=
                    # actions
                    # we need to recompute the state machine after fixing
                    # actions due to the extra reformer machine
                    # this prevents using the latest machine features
                    hypo['state_machine'].machine.reset(
                        hypo['state_machine'].machine.tokens
                    )
                    for action in actions:
                        hypo['state_machine'].machine.update(action)
                    hypo['state_machine'] = hypo['state_machine'].machine

                else:
                    # BART
                    actions_nopos, actions_pos, actions = \
                        post_process_action_pointer_prediction(
                            hypo, self.tgt_dict
                        )

                if self.args.clean_arcs:    # this is 0 by default
                    actions_nopos, actions_pos, actions, invalid_idx = \
                        clean_pointer_arcs(actions_nopos, actions_pos, actions)
                sample_predictions.append({
                    'actions_nopos': actions_nopos,
                    'actions_pos': actions_pos,
                    'actions': actions,
                    'reference': target_tokens,
                    'src_tokens': src_tokens,
                    'sample_id': sample_id,
                    'machine': hypo['state_machine']
                })

            # update sample predictions
            predictions.append(sample_predictions)

        return predictions

    def tokenize(self, sentence):
        assert isinstance(sentence, str)
        return protected_tokenizer(sentence)

    def parse_sentence(self, tokens, **kwargs):
        annotations, decoding_data = self.parse_sentences([tokens], **kwargs)
        return annotations[0], decoding_data[0]

    def parse_sentences(self, batch, batch_size=128, roberta_batch_size=128,
                        gold_amrs=None, force_actions=None, beam=1, jamr=False,
                        no_isi=False):
        """parse a list of sentences.

        Args:
            batch (List[List[str]]): list of tokenized sentences.
            batch_size (int, optional): batch size. Defaults to 128.
            roberta_batch_size (int, optional): RoBerta batch size. Defaults to
            10.
        """
        # check if model is bartsv and force_actions are given
        if 'bartsv' in self.model_args.arch and force_actions is not None:
            raise Exception(
                "The given model is a joint vocabulary model (bartsv) and does not support force actions. Please provide a different model or remove force actions")

        # max batch_size
        if len(batch) < batch_size:
            batch_size = len(batch)
        print("Running on batch size: " + str(batch_size))

        sentences = []
        # The model expects <ROOT> token at the end of the input sentence
        for tokens in batch:
            sentences.append(" ".join(tokens))

        data = self.convert_sentences_to_data(
            sentences,
            batch_size,
            roberta_batch_size,
            gold_amrs=gold_amrs,
            force_actions=force_actions
        )
        data_iterator = self.get_iterator(data, batch_size)
        # Loop over batches of sentences
        completed_machines = {}
        for sample in tqdm(data_iterator, desc='decoding'):
            # move to device
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample

            if 'net_input' not in sample:
                raise Exception("Did not expect empty sample")
                continue

            # parse for this data batch
            predictions = self.parse_batch(sample, to_amr=self.to_amr)

            # collect all annotations
            if not self.to_amr:
                continue

            # collects all sentences x candidates for this batch
            for index, nbest in enumerate(predictions):
                sample_id = nbest[0]['sample_id']
                completed_machines[sample_id] = [c['machine'] for c in nbest]

        # return the AMRs in order
        # TODO: For backwards compatibility squeeze the size 1 list for nbest=1
        annotations = []
        machines = []
        for i in range(0, len(batch)):

            machine_nbest = completed_machines[i]

            if len(machine_nbest) == 1:
                annotations.append(
                    machine_nbest[0].get_annotation(jamr=jamr, no_isi=no_isi)
                )
                machines.append(machine_nbest[0])
            else:
                annotations.append(
                    m.get_annotation(jamr=jamr, no_isi=no_isi)
                    for m in machine_nbest
                )
                machines.append(machine_nbest)

        return annotations, machines

    def parse_docs(self, tok_sentences, **kwargs):

        init_force_actions, doc_sen = get_force_actions(tok_sentences)

        force_actions = [fac+[[]] for fac in init_force_actions]

        annotations, decoding_data = get_sliding_output(
            tok_sentences=doc_sen, parser=self, force_actions=force_actions, **kwargs)

        return annotations, decoding_data


def simple_inspector(machine):
    '''
    print the first machine
    '''
    os.system('clear')
    print(machine)
    input("")


def breakpoint_inspector(machine):
    '''
    call set_trace() on the first machine
    '''
    os.system('clear')
    print(machine)
    set_trace()


def sanity_check_vocabulary_for_alignment(tokens, gold_amrs, tgt_dict):

    for index, amr in enumerate(gold_amrs):
        # if its not in output vocabulary or it can not be copied
        for nname in amr.nodes.values():
            nname = normalize(nname)
            if (
                tgt_dict.index(nname) == tgt_dict.unk()
                and nname not in tokens[index]
            ):
                raise Exception(
                    f'{amr}Has node {nname}, not in vocabulary'
                )

        # if an action with this name is missing
        for (_, edge_label, _) in amr.edges:
            if (
                tgt_dict.index(f'>LA({edge_label})') == tgt_dict.unk()
                and tgt_dict.index(f'>RA({edge_label})') == tgt_dict.unk()
            ):
                raise Exception(
                    f'{amr}Has left-arc for edge {edge_label}, in vocabulary'
                )


def save_multiple_files(args, num_sentences, out_path, string_list):

    if args.nbest == 1 and args.num_samples is None:

        # one single output per input
        with open(out_path, 'w') as fid:
            fid.write('\n'.join(string_list))
            fid.write('\n')
        print(out_path)

    elif args.num_samples is not None:

        # write each corpus samples in a different file
        for j in range(args.num_samples):
            with open(f'{out_path}.{j}', 'w') as fid:
                fid.write('\n'.join([
                    string_list[j + i * args.num_samples]
                    for i in range(num_sentences)
                ]))
                fid.write('\n')
            print(f'{out_path}.{j}')
    else:

        # write each nbest in a different file
        for j in range(args.nbest):
            with open(f'{out_path}.{j}', 'w') as fid:
                fid.write('\n'.join([x[j] for x in string_list]))
                fid.write('\n')
            print(f'{out_path}.{j}')


def load_parser(args, inspector):

    if args.model_name:
        # load from name and optionally seed
        items = args.model_name.split(':')
        model_name = items[0]
        if len(items) > 1:
            seed = items[1]
        else:
            seed = None
        # load from model/config name
        return AMRParser.load(
            model_name,
            seed=seed,
            roberta_cache_path=args.roberta_cache_path,
            inspector=inspector,
            # selected fairseq decoder arguments
            beam=args.beam,
            nbest=args.nbest,
            fp16=args.fp16,
            sampling_topp=args.sampling_topp,
            # this is not, but implies --sampling
            num_samples=args.num_samples
        )
    else:
        # load from checkpoint and files in its folder
        return AMRParser.from_checkpoint(
            args.in_checkpoint,
            roberta_cache_path=args.roberta_cache_path,
            inspector=inspector,
            # selected fairseq decoder arguments
            beam=args.beam,
            nbest=args.nbest,
            fp16=args.fp16,
            sampling_topp=args.sampling_topp,
            # this is not, but implies --sampling
            num_samples=args.num_samples
        )


def run_service(args, parser):
    '''
    Run command line parsing service
    '''

    # set orderd exit
    signal.signal(signal.SIGINT, ordered_exit)
    signal.signal(signal.SIGTERM, ordered_exit)

    while True:
        try:
            sentence = input("Write sentence:\n")
        except EOFError:
            # user pressing C-D
            print("\nStopped by user\n")
            exit(0)

        os.system('clear')

        # skip if sentence is empty
        if not sentence.strip():
            continue

        if args.tokenize:
            # jamr-like tokenization
            tokens = [protected_tokenizer(sentence)[0]]
        else:
            tokens = [sentence.split()]

        if args.in_actions:
            with open(args.in_actions) as fact:
                force_actions = [eval(line.strip()) + [[]] for line in fact]
        else:
            force_actions = None

        if args.sliding:
            gold_amrs = None

            result = get_sliding_output(
                tokens,
                parser,
                gold_amrs,
                force_actions,
                args.window_size,
                args.window_overlap,
                args.batch_size,
                args.roberta_batch_size,
                args.beam,
                args.jamr,
                args.no_isi
            )

        else:

            # duplicate if sampling
            assert args.num_samples is None, \
                "Sampling not supported in --service mode"

            result = parser.parse_sentences(
                tokens,
                batch_size=args.batch_size,
                roberta_batch_size=args.roberta_batch_size,
                force_actions=force_actions,
                beam=args.beam, jamr=args.jamr, no_isi=args.no_isi
            )

        #
        os.system('clear')
        print('\n')
        if args.nbest > 1:
            for candidate in result:
                print(''.join(candidate[0]))
                print()

        else:
            print(''.join(result[0]))


def prepare_data(args, parser):

    force_actions = None
    if args.in_amr:

        # align mode: read input AMR to be aligned
        gold_amrs = read_amr(args.in_amr)
        if args.tokenize:
            # jamr-like tokenization
            assert all(amr.sentence is not None for amr in gold_amrs), \
                "requires field ::snt in Penman meta data"
            tok_sentences = [
                protected_tokenizer(amr.sentence)[0]
                for amr in gold_amrs
            ]
        else:
            tok_sentences = [amr.tokens for amr in gold_amrs]

        # sanity check all node names in vocabulary
        sanity_check_vocabulary_for_alignment(
            tok_sentences, gold_amrs, parser.tgt_dict
        )

    else:

        # normal mode
        gold_amrs = None

        if args.in_doc:
            tok_sentences = []
            docs = read_doc_text(args.in_doc)
            for docsen in docs:
                tok_sen = []
                for sen in docsen:

                    enc = sen.rstrip().encode("ascii", "ignore")

                    newsen = enc.decode()

                    if args.tokenize:
                        tok_sen.append(protected_tokenizer(newsen.rstrip())[0])
                    else:
                        tok_sen.append(newsen.split())
                tok_sentences.append(tok_sen)

        else:
            if args.tokenize:
                # TODO: have tokenized as default
                # jamr-like tokenization
                with open(args.in_tokenized_sentences) as fid:
                    tok_sentences = [
                        protected_tokenizer(sentence.rstrip())[0]
                        for sentence in fid.readlines()
                    ]
            else:
                tok_sentences = read_tokenized_sentences(
                    args.in_tokenized_sentences
                )

    # check for empty sentences
    assert all(s != [''] for s in tok_sentences), "Empty sentences!"

    # sampling needs copy of sentences N times
    if args.num_samples is not None:
        tok_sentences = [
            tsent
            for tsent in tok_sentences
            for _ in range(args.num_samples)
        ]

    return tok_sentences, gold_amrs


def save_data(args, annotations, machines):

    num_sent = len(machines)

    if args.out_amr:
        save_multiple_files(args, num_sent, args.out_amr, annotations)

    # save tokenized sentence
    if args.out_tokens:
        if args.nbest > 1:
            tokens = [
                [' '.join(m.tokens) for m in nbest]
                for nbest in machines
            ]
        else:
            tokens = [' '.join(m.tokens) for m in machines]
        save_multiple_files(args, num_sent, args.out_tokens, tokens)

    # save actions
    if args.out_actions:
        if args.nbest > 1:
            actions = [
                [' '.join(m.action_history) for m in nbest]
                for nbest in machines
            ]
        else:
            actions = [' '.join(m.action_history) for m in machines]
        save_multiple_files(args, num_sent, args.out_actions, actions)


def main():

    # argument handling
    args = argument_parsing()

    # set inspector to use on action loop
    inspector = None
    if args.set_trace:
        inspector = breakpoint_inspector
    if args.step_by_step:
        inspector = simple_inspector

    # load parser
    start = time.time()
    parser = load_parser(args, inspector)
    end = time.time()
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to load parser: {time_secs}')

    if args.service:

        # command line parsing service
        run_service(args, parser)

    else:

        # offline parsing given files
        tok_sentences, gold_amrs = prepare_data(args, parser)
        force_actions = None

        # read constrained decoding forced actions if provided
        if args.in_actions:
            with open(args.in_actions) as fact:
                force_actions = [eval(line.strip()) + [[]] for line in fact]
            assert len(tok_sentences) == len(
                force_actions), "Number of force actions doesn't match the number of sentences"

            # sampling needs copy of force actions N times
            if args.num_samples is not None:
                force_actions = [
                    a
                    for a in force_actions
                    for _ in range(args.num_samples)
                ]

        # TODO: max batch sizes could be computed from max sentence length

        num_sent = len(tok_sentences)
        print(f'Parsing {num_sent} sentences')
        start = time.time()

        if args.in_doc:
            annotations, machines = parser.parse_docs(tok_sentences,
                                                      gold_amrs=gold_amrs,
                                                      window_size=args.window_size,
                                                      window_overlap=args.window_overlap,
                                                      batch_size=args.batch_size,
                                                      roberta_batch_size=args.roberta_batch_size,
                                                      beam=args.beam,
                                                      jamr=args.jamr,
                                                      no_isi=args.no_isi)

        else:

            # normal parsing
            annotations, machines = parser.parse_sentences(
                tok_sentences,
                batch_size=args.batch_size,
                roberta_batch_size=args.roberta_batch_size,
                gold_amrs=gold_amrs,
                force_actions=force_actions,
                beam=args.beam,
                jamr=args.jamr,
                no_isi=args.no_isi
            )

        end = time.time()
        time_secs = timedelta(seconds=float(end-start))
        sents_per_second = num_sent
        if time_secs.seconds > 0:
            sents_per_second = num_sent / time_secs.seconds
        print(f'Total time taken to parse {num_sent} sentences ', end='')
        print(f'beam {args.beam} n-best {args.nbest}')
        print(f'at batch size {args.batch_size}: {time_secs} ', end='')
        print(f'{sents_per_second:.2f} sentences / second')

        # save data resulting from parsing
        save_data(args, annotations, machines)


if __name__ == '__main__':
    main()
