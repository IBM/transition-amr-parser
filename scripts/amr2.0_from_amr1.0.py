import sys


def read_amr_as_raw(file_path):
    with open(file_path) as fid:
        raw_amrs = []
        raw_amr = []
        for line in fid.readlines():
            if line.strip():
                raw_amr.append(line.strip()) 
            else:
                raw_amrs.append(raw_amr)
                raw_amr = []
    return raw_amrs


def write_amr_from_raw(out_file, raw_amrs):
    with open(out_file, 'w') as fid:
        for raw_amr in raw_amrs:
            fid.write('\n'.join(raw_amr))
            fid.write('\n\n')


if __name__ == '__main__':

    in_amr2_path, in_amr1_path, out_amr2_from_amr1_path, out_amr2_minus_amr1_path = sys.argv[1:]

    # read data
    in_amr2 = read_amr_as_raw(in_amr2_path)
    in_amr1 = read_amr_as_raw(in_amr1_path)

    # store by id
    amr2_by_id = {raw_amr[0].split()[2]: raw_amr for raw_amr in in_amr2}
    amr1_by_id = {raw_amr[0].split()[2]: raw_amr for raw_amr in in_amr1}

    # split AMR 2.0
    amr2_from_amr1 = []
    amr2_minus_amr1 = []
    task_labels =  []
    for index, raw_amr in enumerate(in_amr2):
        sentence_id = raw_amr[0].split()[2]
        if sentence_id in amr1_by_id:
            amr2_from_amr1.append(raw_amr)
            task_labels.append('AMR1.0')
        else:
            amr2_minus_amr1.append(raw_amr)
            task_labels.append('AMR2.0')

    # write
    write_amr_from_raw(out_amr2_from_amr1_path, amr2_from_amr1)
    write_amr_from_raw(out_amr2_minus_amr1_path, amr2_minus_amr1)
