# Standalone AMR parser

import os
import math
import torch
from tqdm import tqdm
import copy
import time
from datetime import timedelta

from fairseq import checkpoint_utils, options, tasks, utils
# if you provides model paths separated by : you enable ensembling with no
# further modification
from fairseq.sequence_generator import EnsembleModel
# This is the AMRStateMachine wrapper (for a batch of them)

from transition_amr_parser.stack_transformer.amr_state_machine import (
    StateMachineBatch,
    get_action_indexer,
    machine_generator
)
from transition_amr_parser.stack_transformer.pretrained_embeddings import (
    PretrainedEmbeddings
)
from transition_amr_parser.utils import yellow_font
from transition_amr_parser.io import read_rule_stats, read_sentences
from fairseq.binarizer import Binarizer
from fairseq.tokenizer import tokenize_line
from fairseq.data.language_pair_dataset import collate
from fairseq.models.roberta import RobertaModel


def extract_encoder(sample):
    encoder_input = {
        k: v for k, v in sample['net_input'].items()
        if k not in ['prev_output_tokens']
    }
    # Add dummy memory and memory pos
    encoder_input['memory'] = None
    encoder_input['memory_pos'] = None
    return encoder_input


def get_batch_tensors(sample, source_dictionary, machine_type):

    # auxiliary variables
    encoder_input = extract_encoder(sample)
    src_tokens = encoder_input['src_tokens']
    src_lengths = (
        src_tokens.ne(source_dictionary.eos()) & 
        src_tokens.ne(source_dictionary.pad())
    ).long().sum(dim=1)
    input_size = src_tokens.size()
    bsz = input_size[0]
    
    # max number of steps in episode
    if machine_type == 'NER':
        max_len = int(math.ceil(src_lengths.max().item() * 3))
    elif machine_type == 'AMR':
        max_len = int(math.ceil(src_lengths.max().item() * 10))
    
    # more aux vars
    bos_token = None
    beam_size = 1
    target_actions = src_tokens.new(
        bsz * beam_size, max_len + 2
    ).long().fill_(source_dictionary.pad())
    target_actions[:, 0] = source_dictionary.eos() \
        if bos_token is None else bos_token

    return (
        src_tokens.clone().detach(),
        src_lengths.clone().detach(),
        target_actions
    )


class MachineBatchGenerator():
    """
    Generates a state machine batch precomputing some variables for speed
    """

    def __init__(self, source_dictionary, target_dictionary, machine_type, 
                 machine_rules):

        # Uninitialized batch state machines
        self.state_machine_batch = StateMachineBatch(
            source_dictionary,
            target_dictionary,
            machine_type,
            machine_rules=machine_rules
        )
 
    def new(self, sample):
    
        # auxiliary variables
        encoder_input = extract_encoder(sample)
        src_tokens = encoder_input['src_tokens']
        src_lengths = (
            src_tokens.ne(self.source_dictionary.eos()) & 
            src_tokens.ne(self.source_dictionary.pad())
        ).long().sum(dim=1)
        input_size = src_tokens.size()
        bsz = input_size[0]
        
        # max number of steps in episode
        if self.machine_type == 'NER':
            max_len = int(math.ceil(src_lengths.max().item() * 3))
        elif self.machine_type == 'AMR':
            max_len = int(math.ceil(src_lengths.max().item() * 10))
        
        # more aux vars
        bos_token = None
        beam_size = 1
        target_actions = src_tokens.new(
            bsz * beam_size, max_len + 2
        ).long().fill_(self.source_dictionary.pad())
        target_actions[:, 0] = self.source_dictionary.eos() \
            if bos_token is None else bos_token

        # Initialize state machine and get first states
        # get rules from model folder
        self.state_machine_batch.reset(
            src_tokens[new_order, :].clone().detach(),
            src_lengths[new_order].clone().detach(),
            target_actions.shape[1]
        ) 

        return state_machine_batch, target_actions

class Model():
    """Wrapper around the stack-transformer and state machine"""

    def __init__(self, models, target_dictionary):
        self.temperature = 1.
        self.target_dictionary = target_dictionary
        self.models = models
        self.reset()

    def reset(self):
        # This is to clear the cache of key values, there may be more efficient
        # ways
        self.model = EnsembleModel(self.models)
        # reset cache for encoder
        self.encoder_outs = None
        self.model.eval()

    def precompute_encoder(self, sample):
        """Encoder of the encoder-decoder is fixed and can be precomputed"""
        encoder_input = extract_encoder(sample)
        encoder_outs = self.model.forward_encoder(encoder_input)
        return encoder_outs


    def get_action(self, sample, parser_state, prev_actions):

        # Compute part of the model that does not depend on episode steps
        # (encoder). Cache it for future use
        # precompute encoder for speed
        if self.encoder_outs is None:
            self.encoder_outs = self.precompute_encoder(sample)

        # call model with pre-computed encoder, previous generated actions
        # (tokens) and state machine status
        lprobs, avg_attn_scores = self.model.forward_decoder(
            prev_actions,
            self.encoder_outs,
            parser_state,
            temperature=self.temperature
        )

        # Get most probable action
        if True:
            best_action_indices = lprobs.argmax(dim=1).tolist()
        else:
            # sampling
            best_action_indices = torch.squeeze(lprobs.exp().multinomial(1),1).tolist()
        actions = [self.target_dictionary[i] for i in best_action_indices]
        actions_lprob = [lprobs[0, i] for i in best_action_indices]
        return actions, actions_lprob

class AMRParser():
    def __init__(self, args):
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        self.task = tasks.setup_task(args)
        self.model = self.load_models(args)
#         self.machine_batch_generator = MachineBatchGenerator(
#                 self.task.source_dictionary,
#                 self.task.target_dictionary,
#                 args.machine_type, 
#                 args.machine_rules
#             )

        # Uninitialized batch state machines
        self.state_machine_batch = StateMachineBatch(
            self.task.source_dictionary,
            self.task.target_dictionary,
            args.machine_type,
            machine_rules=args.machine_rules
        )
        # Load RoBERTa
        self.roberta = self.load_roberta(
            name=args.pretrained_embed,
            roberta_cache_path=args.roberta_cache_path,
            roberta_use_gpu=self.use_cuda
        )
        self.embeddings = PretrainedEmbeddings(
            name=args.pretrained_embed,
            bert_layers=args.bert_layers,
            model=self.roberta
        )
        self.parser_batch_size = args.max_sentences
        self.roberta_batch_size = args.roberta_batch_size
        self.left_pad_source = args.left_pad_source
        print("Finished loading models")

    def load_models(self, args):
        models, _ = checkpoint_utils.load_model_ensemble(
                args.path.split(':'),
                arg_overrides=eval(args.model_overrides),
                task=self.task,
                )
        if self.use_cuda:
            print("using GPU for models")
            [m.cuda() for m in models]
        else:
            print("using CPU for models")
        model = Model(models, self.task.target_dictionary)
        return model

    def load_roberta(self, name=None, roberta_cache_path=None, roberta_use_gpu=False):
        if not roberta_cache_path:
            # Load the Roberta Model from torch hub
            roberta = torch.hub.load('pytorch/fairseq', name)
        else:
            roberta = RobertaModel.from_pretrained(roberta_cache_path, checkpoint_file='model.pt')
        roberta.eval()
        if roberta_use_gpu:
            roberta.cuda()
        return roberta

    def get_bert_features_batched(self, sentences, batch_size):
        bert_data = []
        for i in tqdm(range(0, math.ceil(len(sentences)/batch_size)), desc='roberta'):
            batch = sentences[i * batch_size: i * batch_size + batch_size]
            batch_data = self.embeddings.extract_batch(batch)
            for i in range(0, len(batch)):
                bert_data.append((copy.deepcopy(batch_data["word_features"][i]), copy.deepcopy(batch_data["wordpieces_roberta"][i]), copy.deepcopy(batch_data["word2piece_scattered_indices"][i])))
        print(len(bert_data))
        assert len(bert_data) == len(sentences)
        return bert_data

    def get_token_ids(self, sentence):
        return self.task.source_dictionary.encode_line(line=sentence,
                                                line_tokenizer=tokenize_line,
                                                add_if_not_exist=False,
                                                append_eos=False,
                                                reverse_order=False)

    def convert_sentences_to_data(self, sentences, batch_size):
        roberta_features = self.get_bert_features_batched(sentences, self.roberta_batch_size)
        
        data = []
        for index, sentence in enumerate(sentences):
            ids = self.get_token_ids(sentence)
            word_features, wordpieces_roberta, word2piece_scattered_indices = roberta_features[index]
            data.append({
                'id': index,
                'source': ids,
                'source_fix_emb': word_features,
                'src_wordpieces': wordpieces_roberta, 
                'src_wp2w': word2piece_scattered_indices,
                'orig_tokens': tokenize_line(sentence)
            })
        return data

    def get_iterator(self, samples, batch_size):
        batches = []
        for i in range(0, math.ceil(len(samples)/batch_size)):
            sample = samples[i * batch_size: i * batch_size + batch_size]
            batch = collate(
                sample, pad_idx=self.task.source_dictionary.pad(), eos_idx=self.task.source_dictionary.eos(),
                left_pad_source=self.left_pad_source,
                state_machine=False
            )
            batches.append(batch)
        return batches

    def parse_sentences(self, batch):
        sentences = []
        # The model expects <ROOT> token at the end of the input sentence
        for tokens in batch:
            if tokens[-1] != "<ROOT>":
                tokens.append("<ROOT>")
            # TODO: We are joining with tabs since the model was trained this way. 
            # Change it to whitespace once we update to using the model trained with whitespace separated text
            sentences.append(" ".join(tokens))

        if len(batch) < self.parser_batch_size:
            batch_size = len(batch)
        else:
            batch_size = self.parser_batch_size
        print("Running on batch size: " + str(batch_size))

        data = self.convert_sentences_to_data(sentences, batch_size)
        data_iterator = self.get_iterator(data, batch_size)
        # final annotations
        amr_annotations = {}

        for sample in tqdm(data_iterator, desc='decoding'):

            # step, sample, state_machine_batch, tokens = state
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample 

            # Get input tensors and preallocate space for output tensors
            # TODO: This could be inside StateMachineBatch but we need to be
            # coherent with fairseq/generate.py
            src_tokens, src_lengths, target_actions = get_batch_tensors(
                sample,
                self.state_machine_batch.src_dict,
                self.state_machine_batch.machine_type,
            )
            # Initialize state machine and get first states
            # get rules from model folder
            self.state_machine_batch.reset(
                src_tokens,
                src_lengths,
                target_actions.shape[1]
            ) 

            # Reset model. This is to clean up the key/value cache in the decoder
            # and the encoder cache. There may be more efficient ways.
            self.model.reset()

            # Loop over actions until all machines finish
            done = True
            time_step = 0
            while any(not m.is_closed for m in self.state_machine_batch.machines):

                # DEBUG
                # os.system('clear')
                # print(self.state_machine_batch.machines[2])
                # import pdb;pdb.set_trace()

                # Get target masks from machine state
                logits_indices, logits_mask = self.state_machine_batch.get_active_logits()
                parser_state = (
                    self.state_machine_batch.memory[:, :, :time_step + 1].clone(),
                    self.state_machine_batch.memory_pos[:, :, :time_step + 1].clone(),
                    logits_mask,
                    logits_indices
                )

                # Get most probably action from the model for each sentence given
                # input data, previous actions and state machine state
                actions, log_probs = self.model.get_action(
                    sample,
                    parser_state,
                    target_actions[:, :time_step + 1].data,
                )

                # act on the state machine batch
                # Loop over paralele machines in the batch
                for machine_idx, machine in enumerate(self.state_machine_batch.machines):

                    # Emergency stop. If we reach maximum action number, force
                    # stop the machine
                    if (
                        time_step + 2 == target_actions.shape[1] and
                        not machine.is_closed
                    ):
                        msg = (
                            f'machine {machine_idx} not closed at step '
                            '{time_step}'
                        )
                        print(yellow_font(msg))
                        action = 'CLOSE'
                    else:
                        action = actions[machine_idx]

                    # update state machine
                    machine.update(action)
                    # update list of previous actions
                    action_idx = self.task.target_dictionary.index(action)
                    target_actions[machine_idx, time_step + 1] = action_idx

                # update counters and recompute masks
                time_step +=1
                self.state_machine_batch.step_index += 1
                self.state_machine_batch.update_masks()

            # collect all annotations
            ids = sample["id"].detach().cpu().tolist()
            for index, id in enumerate(ids):
                amr_annotations[id] = self.state_machine_batch.machines[index].get_annotations()

        # return the AMRs
        result = []
        for i in range(0,len(batch)):
            result.append(amr_annotations[i])

        return result
