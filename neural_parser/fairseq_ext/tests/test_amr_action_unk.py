from collections import Counter

from tqdm import tqdm

from neural_parser.fairseq_ext.data.amr_bpe import AMRActionBPEEncoder, AMRActionBartDictionary
from neural_parser.fairseq_ext.amr_reform.o10_action_reformer_subtok import AMRActionReformerSubtok
from neural_parser.transition_amr_parser.amr_machine import AMRStateMachine


if __name__ == '__main__':
    # file paths
    encoder_json_path = 'DATA/gpt2_bpe/encoder.json'
    vocab_bpe_path = 'DATA/gpt2_bpe/vocab.bpe'
    dict_txt_path = 'DATA/gpt2_bpe/dict.txt'
    node_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.nodes'
    others_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.others'

    split = 'train'
    en_file = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/{split}.tokens'
    actions_file = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/{split}.actions'
    machine_config = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/machine_config.json'

    # # build the bpe encoder
    # act_bpe = AMRActionBPEEncoder.build_bpe_encoder(encoder_json_path,    # or None to use cached
    #                                                 vocab_bpe_path,    # or None to use cached
    #                                                 # add new symbols
    #                                                 node_freq_min=5,
    #                                                 node_file_path=node_file_path,
    #                                                 others_file_path=others_file_path
    #                                                 )

    # actions = 'SHIFT SHIFT clear-06 ROOT SHIFT SHIFT thing prepare-01 >RA(:ARG1-of) SHIFT prior-to >RA(:time) ' \
    #     'SHIFT SHIFT SHIFT COPY >RA(:op1) SHIFT SHIFT - SHIFT construct-01 >LA(:polarity) >LA(:ARG1) >RA(:ARG1) ' \
    #     'SHIFT SHIFT SHIFT base-02 >RA(:ARG1-of) SHIFT SHIFT SHIFT COPY SHIFT COPY >LA(:mod) SHIFT simulate-01 >LA(:ARG1) >RA(:ARG2) SHIFT SHIFT'
    # bpe_token_ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index = act_bpe.encode_actions(actions)

    # breakpoint()

    # build the action dictionary
    act_dict = AMRActionBartDictionary(dict_txt_path,    # or None to use cached
                                       node_freq_min=5,
                                       node_file_path=node_file_path,
                                       others_file_path=others_file_path)

    # ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index = act_dict.encode_actions(actions)

    # breakpoint()

    # print(act_dict.decode_actions(ids))

    # check for unk symbol in the data
    machine = AMRStateMachine.from_config(machine_config)

    replaced = Counter()
    current_unk = []

    def replaced_consumer(word, idx):
        if idx == act_dict.unk_index and word != act_dict.unk_word:
            replaced.update([word])
            current_unk.append(word)

    with open(en_file, 'r') as f, open(actions_file, 'r') as g:
        for tokens, actions in tqdm(zip(f, g)):
            if tokens.strip():
                tokens = tokens.strip().split('\t')
                actions = actions.strip().split('\t')

                if actions[-1] != 'CLOSE':
                    actions = actions.copy()
                    actions.append('CLOSE')

                actions_states = AMRActionReformerSubtok.reform_actions_and_get_states(tokens, actions,
                                                                                       act_dict, machine)
                v = actions_states['actions_nopos_out']

                current_unk = []

                ids = act_dict.encode_line(
                    line=[act if act != 'CLOSE' else act_dict.eos_word for act in v],
                    line_tokenizer=lambda x: x,    # already tokenized
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=False,
                    reverse_order=False
                    )

                if current_unk:
                    print(replaced)
                    print(current_unk)
                    print(actions)
                    breakpoint()
