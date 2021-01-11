from transition_amr_parser.apt_amr_parser import AMRParser


in_checkpoint = 'EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/checkpoint_wiki-smatch_top5-avg.pt'


if __name__ == '__main__':
    parser = AMRParser.from_checkpoint(in_checkpoint)
    while True:
        sentence = input('\nPlease enter a sentence to parse ("exit" to exit):\n')
        if sentence == 'exit':
            break
        batch = [sentence.split()]
        print('\n' + '-' * 30 + ' parsing ' + '-' * 30)
        annotations, predictions = parser.parse_sentences(batch)
        print('\n' + '*' * 30 + ' Resulted AMR graph: \n')
        print(annotations[0])
        # breakpoint()
