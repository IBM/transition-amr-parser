from neural_parser.fairseq_ext.data.amr_bpe import AMRActionBPEEncoder, AMRActionBartDictionary


if __name__ == '__main__':
    # file paths
    encoder_json_path = 'DATA/gpt2_bpe/encoder.json'
    vocab_bpe_path = 'DATA/gpt2_bpe/vocab.bpe'
    dict_txt_path = 'DATA/gpt2_bpe/dict.txt'
    node_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.nodes'
    others_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.others'

    # build the bpe encoder
    act_bpe = AMRActionBPEEncoder.build_bpe_encoder(encoder_json_path,    # or None to use cached
                                                    vocab_bpe_path,    # or None to use cached
                                                    # add new symbols
                                                    node_freq_min=5,
                                                    node_file_path=node_file_path,
                                                    others_file_path=others_file_path
                                                    )

    actions = 'SHIFT SHIFT clear-06 ROOT SHIFT SHIFT thing prepare-01 >RA(:ARG1-of) SHIFT prior-to >RA(:time) ' \
        'SHIFT SHIFT SHIFT COPY >RA(:op1) SHIFT SHIFT - SHIFT construct-01 >LA(:polarity) >LA(:ARG1) >RA(:ARG1) ' \
        'SHIFT SHIFT SHIFT base-02 >RA(:ARG1-of) SHIFT SHIFT SHIFT COPY SHIFT COPY >LA(:mod) SHIFT simulate-01 >LA(:ARG1) >RA(:ARG2) SHIFT SHIFT'
    bpe_token_ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index = act_bpe.encode_actions(actions)

    breakpoint()

    # build the action dictionary
    act_dict = AMRActionBartDictionary(dict_txt_path,    # or None to use cached
                                       node_freq_min=5,
                                       node_file_path=node_file_path,
                                       others_file_path=others_file_path)

    ids, bpe_tokens, tok_to_subtok_start, subtok_origin_index = act_dict.encode_actions(actions)

    breakpoint()

    print(act_dict.decode_actions(ids))
