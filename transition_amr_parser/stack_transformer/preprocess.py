from collections import Counter
import json
import numpy as np
import torch
from tqdm import tqdm
import os
from fairseq.data import indexed_dataset
from transition_amr_parser.io import read_rule_stats
from transition_amr_parser.stack_transformer.amr_state_machine import (
    machine_generator,
    get_word_states,
    yellow_font,
    get_action_indexer
)
from transition_amr_parser.stack_transformer.pretrained_embeddings import (
    PretrainedEmbeddings
)


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def make_binary_stack(args, target_vocab, input_prefix, output_prefix, eos_idx, pad_idx, mask_predicates=False, allow_unk=False, tokenize=None):

    assert tokenize

    # involved files
    # for debug
    input_senteces = input_prefix + '.en'
    input_actions = input_prefix + '.actions'

    # The AMR state machine allways expects rules
    if args.machine_type == 'AMR':

        assert args.machine_rules and os.path.isfile(args.machine_rules), \
            f'Missing {args.machine_rules}'

        # Read rules
        train_rule_stats = read_rule_stats(args.machine_rules)
        actions_by_stack_rules = train_rule_stats['possible_predicates']
    else:
        actions_by_stack_rules = None
        
    action_indexer = get_action_indexer(target_vocab.symbols)

    # initialize indices for each of variables
    # memory (stack, buffer, dead) (position in memory)
    stack_buffer_names = ['memory', 'memory_pos']
    # FIXME: These values are hard-coded elsewhere in code
    state_indices = [3, 4, 5]
    assert eos_idx not in state_indices, "Can not reuse EOS index"
    assert pad_idx not in state_indices, "Can not reuse PAD index"
    indexed_data = {}
    for name in stack_buffer_names:
        indexed_data[name] = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, name, "bin"),
            impl=args.dataset_impl,
        )

    if mask_predicates:
        # mask of target predictions 
        masks_path = dataset_dest_file(args, output_prefix, 'target_masks', "bin")
        indexed_target_masks = indexed_dataset.make_builder(
            masks_path,
            impl=args.dataset_impl,
        )

        # active indices 
        active_logits_path = dataset_dest_file(args, output_prefix, 'active_logits', "bin")
        indexed_active_logits = indexed_dataset.make_builder(
            active_logits_path,
            impl=args.dataset_impl,
        )

    # Returns function that generates initialized state machines given
    # sentence 
    get_new_state_machine = machine_generator(actions_by_stack_rules, entity_rules=args.entity_rules)

    num_sents = 0
    missing_actions = Counter()
    with open(input_actions, 'r') as fid_act, \
         open(input_senteces, 'r') as fid_sent:

        # Loop over sentences
        for sentence in tqdm(fid_sent):

            # Get actions, tokens
            sent_tokens = tokenize(sentence)
            sent_actions = tokenize(fid_act.readline())

            # intialize state machine batch for size 1
            state_machine = get_new_state_machine(
                sent_tokens,
                machine_type=args.machine_type
            )

            # collect target and source masks
            sent_data = {}
            for name in stack_buffer_names:
                sent_data[name] = []

            shape = (len(sent_actions), len(target_vocab.symbols))
            logits_mask = np.zeros(shape)
            active_logits = set()
            for action_idx, gold_action in enumerate(sent_actions):

                # active logits for this action
                if mask_predicates:

                    # Get total valid actions by expanding base ones
                    valid_actions, invalid_actions = state_machine.get_valid_actions()
                    valid_action_idx = (
                        action_indexer(valid_actions) 
                        - action_indexer(invalid_actions)
                    )

                    # if action is missing add it and count it
                    if gold_action in target_vocab.symbols:
                        gold_action_index = target_vocab.symbols.index(gold_action) 
                    else:
                        gold_action_index = target_vocab.symbols.index('<unk>') 
                    if gold_action_index not in valid_action_idx:
                        valid_action_idx.add(gold_action_index)
                        missing_actions.update([gold_action])

                    # if length 1 add pad to avoid deltas during learning
                    if len(valid_action_idx) == 1:
                        valid_action_idx.add(pad_idx)

                    # append number of nodes to regain matrix
                    logits_mask[action_idx, list(valid_action_idx)] = 1
                    active_logits |= valid_action_idx

                # stack and buffer 
                memory, memory_pos = get_word_states(
                    state_machine,
                    sent_tokens,
                    indices=state_indices
                )

                # word states
                sent_data['memory'].append(torch.Tensor(memory))
                # note we use position 0 for reduced words
                sent_data['memory_pos'].append(
                    torch.Tensor(memory_pos)
                )

                # Update machine
                state_machine.applyAction(gold_action)

            for name in stack_buffer_names:
                # note that data needs to be stores as a 1d array
                indexed_data[name].add_item(
                    torch.stack(sent_data[name]).view(-1)
                )

            # valid nodes
            if mask_predicates:
                active_logits = list(active_logits)
                # reduce size to active items
                logits_mask = logits_mask[:, active_logits]
                indexed_target_masks.add_item(
                    torch.Tensor(logits_mask).view(-1)
                )

                # active indices 
                indexed_active_logits.add_item(torch.Tensor(
                    active_logits
                ))

            # update number of sents
            num_sents += 1
            if not num_sents % 100:
                print("\r%d sentences" % num_sents, end = '')

        print("")

    # close indexed data files
    for name in stack_buffer_names:
        output_file_idx = dataset_dest_file(args, output_prefix, name, "idx")
        indexed_data[name].finalize(output_file_idx)
    # close valid action mask
    if mask_predicates:
        target_mask_idx = dataset_dest_file(args, output_prefix, 'target_masks', "idx")
        indexed_target_masks.finalize(target_mask_idx)

        # active indices 
        active_logits_idx = dataset_dest_file(args, output_prefix, 'active_logits', "idx")
        indexed_active_logits.finalize(active_logits_idx)

    # inform about mssing actions
    if missing_actions: 
        print(yellow_font("There were missing actions"))
        print(missing_actions)


def make_binary_bert_features(args, input_prefix, output_prefix, eos_idx, pad_idx, tokenize):

    # Load pretrained embeddings extractor
    pretrained_embeddings = PretrainedEmbeddings(
        args.pretrained_embed,
        args.bert_layers
    )

    # will store pre-extracted BERT layer
    indexed_data = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, 'en.bert', "bin"),
        impl=args.dataset_impl,
        dtype=np.float32
    )

    # will store wordpieces and wordpiece to word mapping
    indexed_wordpieces = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, 'en.wordpieces', "bin"),
        impl=args.dataset_impl,
    )

    indexed_wp2w = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, 'en.wp2w', "bin"),
        impl=args.dataset_impl,
    )

    num_sents = 0
    input_file = input_prefix + '.en'

    with open(input_file, 'r') as fid:
        for sentence in fid:

            # we only have tokenized data so we feed whitespace separated
            # tokens
            sentence = " ".join(tokenize(str(sentence).rstrip()))

            # extract embeddings, average them per token and return
            # wordpieces anyway
            word_features, worpieces_roberta, word2piece = \
                pretrained_embeddings.extract(sentence)

            # note that data needs to be stored as a 1d array. Also check
            # that number nof woprds matches with embedding size
            assert word_features.shape[1] == len(sentence.split())
            indexed_data.add_item(word_features.cpu().view(-1))

            # just store the wordpiece indices, ignore BOS/EOS tokens
            indexed_wordpieces.add_item(worpieces_roberta)
            indexed_wp2w.add_item(
                get_scatter_indices(word2piece, reverse=True)
            )

            # udpate number of sents
            num_sents += 1
            if not num_sents % 100:
                print("\r%d sentences" % num_sents, end = '')
        print("")

    # close indexed data files
    indexed_data.finalize(
        dataset_dest_file(args, output_prefix, 'en.bert', "idx")
    )

    indexed_wordpieces.finalize(
        dataset_dest_file(args, output_prefix, 'en.wordpieces', "idx")
    )
    indexed_wp2w.finalize(
        dataset_dest_file(args, output_prefix, 'en.wp2w', "idx")
    )


def make_masks(args, target_vocab, input_prefix, output_prefix, eos_idx, pad_idx, mask_predicates=False, allow_unk=False, tokenize=None):
    assert tokenize
    if args.dataset_impl == "raw":
        raise NotImplementedError("only binary supported for now")
    else:
        make_binary_stack(args, target_vocab, input_prefix, output_prefix, eos_idx, pad_idx, mask_predicates=mask_predicates, allow_unk=allow_unk, tokenize=tokenize)


def make_state_machine(args, src_dict, tgt_dict, tokenize=None):
    '''
    Makes BERT features and source and target masks
    '''

    assert tokenize

    if args.trainpref:
        make_binary_bert_features(args, args.trainpref, "train", src_dict.eos_index, src_dict.pad_index, tokenize)
        make_masks(args, tgt_dict, args.trainpref, "train", tgt_dict.eos_index, tgt_dict.pad_index, mask_predicates=True, tokenize=tokenize)

    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            make_binary_bert_features(args, validpref, outprefix, src_dict.eos_index, src_dict.pad_index, tokenize)
            make_masks(args, tgt_dict, validpref, outprefix, tgt_dict.eos_index, tgt_dict.pad_index, mask_predicates=True, allow_unk=True, tokenize=tokenize)

    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            make_binary_bert_features(args, testpref, outprefix, src_dict.eos_index, src_dict.pad_index, tokenize)
            make_masks(args, tgt_dict, testpref, outprefix, tgt_dict.eos_index, tgt_dict.pad_index, mask_predicates=True, allow_unk=True, tokenize=tokenize)


def get_scatter_indices(word2piece, reverse=False):
    if reverse:
        indices = range(len(word2piece))[::-1]
    else:
        indices = range(len(word2piece))
    # we will need as well the wordpiece to word indices
    wp_indices = [
        [index] * (len(span) if isinstance(span, list) else 1)
        for index, span in zip(indices, word2piece)
    ]
    wp_indices = [x for span in wp_indices for x in span]
    return  torch.tensor(wp_indices)
