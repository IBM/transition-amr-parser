# Standalone AMR parser from an existing trained APT model

import os
import time
import math
import copy
import signal
import argparse
from datetime import timedelta

from ipdb import set_trace
from tqdm import tqdm
import torch
from fairseq import checkpoint_utils, utils
from fairseq.tokenizer import tokenize_line
from fairseq.models.bart import BARTModel

from fairseq_ext import options
from fairseq_ext.extract_bart.sentence_encoding import SentenceEncodingBART
from fairseq_ext.extract_bart.binarize_encodings import get_scatter_indices
from fairseq_ext.data.amr_action_pointer_dataset import collate
from transition_amr_parser.io import read_tokenized_sentences, read_amr
from fairseq_ext.utils import (
    post_process_action_pointer_prediction,
    post_process_action_pointer_prediction_bartsv,
    clean_pointer_arcs
)
from fairseq_ext.utils_import import import_user_module
from transition_amr_parser.io import protected_tokenizer
from transition_amr_parser.amr import normalize


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
        '--in-amr',
        type=str,
        help='AMR in Penman format to align'
    )
    parser.add_argument(
        '--tokenize',
        action='store_true',
        help='Tokenize with a jamr-like tokenizer input sentences or AMR'
    )
    parser.add_argument(
        '--beam',
        type=int,
        default=1,
        help='Beam decoding size'
    )
    parser.add_argument(
        '--service',
        action='store_true',
        help='Prompt user for sentences'
    )
    parser.add_argument(
        '-c', '--in-checkpoint',
        type=str,
        required=True,
        help='one fairseq model checkpoint (or various, separated by :)'
    )
    parser.add_argument(
        '-o', '--out-amr',
        type=str,
        help='File to store AMR in PENNMAN format'
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
    args = parser.parse_args()

    # sanity checks
    assert (
        bool(args.in_tokenized_sentences) or bool(args.in_amr)
    ) or bool(args.service), \
        "Must either specify --in-tokenized-sentences or set --service"

    return args


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)


def load_models_and_task(args, use_cuda, task=None):
    # if `task` is not provided, it will be from the saved model args
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
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
    def from_checkpoint(cls, checkpoint, dict_dir=None,
                        roberta_cache_path=None, fp16=False,
                        inspector=None, beam=1):
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
        if dict_dir is not None:
            args.model_overrides = f'{{"data": "{dict_dir}"}}'
            # otherwise, the default dict folder is read from the model args
        use_cuda = torch.cuda.is_available() and not args.cpu
        models, model_args, task = load_models_and_task(
            args, use_cuda, task=None
        )
        # overload some arguments
        args.beam = beam

        # load pretrained Roberta model for source embeddings
        if model_args.pretrained_embed_dim == 768:
            pretrained_embed = 'bart.base'
        elif model_args.pretrained_embed_dim == 1024:
            pretrained_embed = 'bart.large'
        else:
            raise ValueError

        model_folder = os.path.dirname(checkpoint.split(':')[0])
        config_data_path = None
        for dfile in os.listdir(model_folder):
            if dfile.startswith('config.sh'):
                config_data_path = os.path.join(model_folder, dfile)
                break
        assert config_data_path is not None, \
            'data configuration file not found'

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
                                  roberta_batch_size, gold_amrs=None):

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
                'gold_amr': None if gold_amrs is None else gold_amrs[index]
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

            batches.append(batch)
        return batches

    def parse_batch(self, sample, to_amr=True):
        hypos = self.task.inference_step(
            self.generator, self.models, sample, self.args, prefix_tokens=None
        )
        assert self.args.nbest == 1, \
            'Currently we only support outputing the top predictions'

        predictions = []
        for i, sample_id in enumerate(sample['id'].tolist()):
            src_tokens = sample['src_sents'][i]
            target_tokens = None

            for j, hypo in enumerate(hypos[i][:self.args.nbest]):
                # args.nbest is default 1, i.e. saving only the top predictions
                if 'bartsv' in self.model_args.arch:
                    # shared vocabulary BART
                    actions_nopos, actions_pos, actions = \
                        post_process_action_pointer_prediction_bartsv(
                            hypo, self.tgt_dict
                        )
                else:
                    # BART
                    actions_nopos, actions_pos, actions = \
                        post_process_action_pointer_prediction(
                            hypo, self.tgt_dict
                        )

                if self.args.clean_arcs:    # this is 0 by default
                    actions_nopos, actions_pos, actions, invalid_idx = \
                        clean_pointer_arcs(actions_nopos, actions_pos, actions)
                predictions.append({
                    'actions_nopos': actions_nopos,
                    'actions_pos': actions_pos,
                    'actions': actions,
                    'reference': target_tokens,
                    'src_tokens': src_tokens,
                    'sample_id': sample_id,
                    'machine': hypo['state_machine']
                })

        return predictions

    def parse_sentences(self, batch, batch_size=128, roberta_batch_size=128,
                        gold_amrs=None, beam=1):
        """parse a list of sentences.

        Args:
            batch (List[List[str]]): list of tokenized sentences.
            batch_size (int, optional): batch size. Defaults to 128.
            roberta_batch_size (int, optional): RoBerta batch size. Defaults to
            10.
        """
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
            gold_amrs=gold_amrs
        )
        data_iterator = self.get_iterator(data, batch_size)

        # Loop over batches of sentences
        amr_annotations = {}
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

            # FIXME: Entropic
            for index, pred_dict in enumerate(predictions):
                sample_id = pred_dict['sample_id']
                amr_annotations[sample_id] = \
                    pred_dict['machine'].get_annotation()

        # return the AMRs in order
        results = []
        for i in range(0, len(batch)):
            results.append(amr_annotations[i])

        return results, predictions


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
    parser = AMRParser.from_checkpoint(
        args.in_checkpoint,
        roberta_cache_path=args.roberta_cache_path,
        inspector=inspector,
        beam=args.beam
    )
    end = time.time()
    time_secs = timedelta(seconds=float(end-start))
    print(f'Total time taken to load parser: {time_secs}')

    # TODO: max batch sizes could be computed from max sentence length
    if args.service:

        # set orderd exit
        signal.signal(signal.SIGINT, ordered_exit)
        signal.signal(signal.SIGTERM, ordered_exit)

        while True:
            sentence = input("Write sentence:\n")
            os.system('clear')
            if not sentence.strip():
                continue

            if args.tokenize:
                # jamr-like tokenization
                tokens = [protected_tokenizer(sentence)[0]]
            else:
                tokens = [sentence.split()]

            result = parser.parse_sentences(
                tokens,
                batch_size=args.batch_size,
                roberta_batch_size=args.roberta_batch_size,
                beam=args.beam
            )
            #
            os.system('clear')
            print('\n')
            print(''.join(result[0]))

    else:

        if args.in_amr:
            gold_amrs = read_amr(args.in_amr)
            if args.tokenize:
                # jamr-like tokenization
                assert all(amr.sentence is not None for amr in gold_amrs)
                tokenized_sentences = [
                    protected_tokenizer(amr.sentence)[0]
                    for amr in gold_amrs
                ]
            else:
                tokenized_sentences = [amr.tokens for amr in gold_amrs]

            # sanity check all node names in vocabulary
            sanity_check_vocabulary_for_alignment(
                tokenized_sentences, gold_amrs, parser.tgt_dict
            )

        else:
            gold_amrs = None
            tokenized_sentences = read_tokenized_sentences(
                args.in_tokenized_sentences
            )
            if args.tokenize:
                # jamr-like tokenization
                tokenized_sentences = [
                    protected_tokenizer(sentence)[0]
                    for sentence in tokenized_sentences
                ]

        # Parse sentences
        result = parser.parse_sentences(
            tokenized_sentences,
            batch_size=args.batch_size,
            roberta_batch_size=args.roberta_batch_size,
            gold_amrs=gold_amrs,
            beam=args.beam
        )

        with open(args.out_amr, 'w') as fid:
            fid.write('\n'.join(result[0]))


if __name__ == '__main__':
    main()
