from fairseq_ext.data.amr_bpe import AMRActionBPEEncoder


if __name__ == '__main__':
    # file paths
    encoder_json_path = 'DATA/gpt2_bpe/encoder.json'
    vocab_bpe_path = 'DATA/gpt2_bpe/vocab.bpe'
    node_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.nodes'
    others_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.others'

    # build the bpe encoder
    act_bpe = AMRActionBPEEncoder.build_bpe_encoder(encoder_json_path,
                                                    vocab_bpe_path,
                                                    # add new symbols
                                                    node_freq_min=5,
                                                    node_file_path=node_file_path,
                                                    others_file_path=others_file_path
                                                    )

    breakpoint()

    