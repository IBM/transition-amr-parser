from transition_amr_parser.apt_amr_parser import AMRParser


in_checkpoint = 'DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_wiki.smatch_top5-avg.pt'


if __name__ == '__main__':
    parser = AMRParser.from_checkpoint(in_checkpoint)
    annotations, predictions = parser.parse_sentences([['The', 'boy', 'travels', 'to', 'New', 'York'], ['He', 'visits', 'places']])
    print(annotations[0])
    breakpoint()
