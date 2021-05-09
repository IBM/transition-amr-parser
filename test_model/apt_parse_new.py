from transition_amr_parser.apt_amr_parser import AMRParser


in_checkpoint = 'EXP/exp_o10_bart-large_act-pos_vmask1_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb/models_ep100_seed42_fp16-lr0.0001-mt2048x4-wm4000-dp0.2/models_ep120_seed42/checkpoint_wiki-smatch_top5-avg.pt'


if __name__ == '__main__':
    parser = AMRParser.from_checkpoint(in_checkpoint)
    annotations, predictions = parser.parse_sentences([['The', 'boy', 'travels', 'to', 'New', 'York'], ['He', 'visits', 'places']])
    print(annotations[0])
    breakpoint()
