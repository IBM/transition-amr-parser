import argparse

def amr_add_id(file_path,file_path_id):

    with open(file_path_id) as fid1:
        ids_list = []
        for line in fid1.readlines():
            if '# ::id ' in line:
                ids_list.append(line)


    with open(file_path) as fid2:
        raw_amr = []
        ids_idx = 0
        for line in fid2.readlines():
            if '::tok' in line :
                raw_amr.append(ids_list[ids_idx])
                ids_idx+=1
            raw_amr.append(line)
        assert len(ids_list)==ids_idx

    with open(file_path.rstrip('.txt')+'_id-added.txt','w') as fid3:
        for line in raw_amr:
            fid3.write(line)
                
        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Produces oracle sequences given AMR alignerd to sentence'
    )
    # Single input parameters
    parser.add_argument(
        "--in-aligned-amr",
        help="In file containing AMR in penman format AND isi alignments ",
        type=str,
        default='DATA/AMR3.0/aligned/cofill_isi/train.txt'
    )

    parser.add_argument(
        "--amr-with-id",
        help="add id to --in-aligned-amr using the file given",
        type=str,
        default='DATA/AMR3.0/aligned/cofill/train.txt'
    )
    args = parser.parse_args()

    amr_add_id(args.in_aligned_amr,args.amr_with_id)
