from tqdm import tqdm
import argparse
import re
import subprocess
from ipdb import set_trace
from transition_amr_parser.io import read_blocks
# this requires local smatch installed --editable and touch smatch/__init__.py
from smatch.smatch import get_best_match, compute_f
from smatch import amr
import smatch


def format_penman(x):
    # remove comments
    lines = '\n'.join([x for x in x.split('\n') if x and x[0] != '#'])
    # remove ISI alignments if any
    lines = re.sub(r'\~[0-9]+', '', lines)
    return lines.replace('\n', '')


def protected_read(penman, index, prefix):

    fpenman = format_penman(penman)

    damr = amr.AMR.parse_AMR_line(fpenman)
    # stop if reading failed
    if damr is None:
        set_trace(context=30)
        print()

    # original ids
    ids = [x[1] for x in damr.get_triples()[0]]

    # prefix name as per original code
    damr.rename_node(prefix)

    return damr, ids


def vimdiff(penman1, penman2):

    def write(file_name, content):
        with open(file_name, 'w') as fid:
            fid.write(content)

    write('_tmp1', penman1)
    write('_tmp2', penman2)
    subprocess.call(['vimdiff', '_tmp1', '_tmp2'])


def main(args):

    # read files
    gold_penmans = read_blocks(args.in_reference_amr, return_tqdm=False)
    oracle_penmans = read_blocks(args.in_amr, return_tqdm=False)

    # set global in module
    smatch.smatch.iteration_num = args.r + 1

    assert len(gold_penmans) == len(oracle_penmans)

    statistics = []
    sentence_smatch = []
    node_id_maps = []
    for index in tqdm(range(len(oracle_penmans))):

        # read and format. Keep original ids for later reconstruction
        amr1, ids_a = protected_read(gold_penmans[index], index, "a")
        amr2, ids_b = protected_read(oracle_penmans[index], index, "b")

        # use smatch code to align nodes
        (instance1, attributes1, relation1) = amr1.get_triples()
        (instance2, attributes2, relation2) = amr2.get_triples()
        best_mapping, best_match_num = get_best_match(
            instance1, attributes1, relation1,
            instance2, attributes2, relation2,
            "a", "b"
        )
        # IMPORTANT: Reset cache
        smatch.smatch.match_triple_dict = {}
        # use smatch code to compute partial scores
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)

        # accumulate statistics for corpus-level normalization
        statistics.append([best_match_num, test_triple_num, gold_triple_num])
        # store sentence-level normalized score
        score = compute_f(best_match_num, test_triple_num, gold_triple_num)
        sentence_smatch.append(score)

        # store alignments
        sorted_ids_b = [ids_b[i] for i in best_mapping]
        best_id_map = dict(zip(ids_a, sorted_ids_b))
        node_id_maps.append(best_id_map)

        # stop if score is not perfect
        if (
            args.stop_if_different
            and (
                best_match_num != gold_triple_num
                or test_triple_num != gold_triple_num
            )
        ):
            print(gold_penmans[index])
            print(oracle_penmans[index])
            vimdiff(gold_penmans[index], oracle_penmans[index])
            print()

    # TODO: Save scores and alignments

    # final score
    best_match_num = sum([t[0] for t in statistics])
    test_triple_num = sum([t[1] for t in statistics])
    gold_triple_num = sum([t[2] for t in statistics])
    corpus_score = compute_f(best_match_num, test_triple_num, gold_triple_num)
    print(f'Smatch: {corpus_score[2]:.2f}')


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    parser.add_argument(
        "--in-reference-amr",
        help="file with reference AMRs in penman format",
        type=str
    )
    parser.add_argument(
        "--in-amr",
        help="file with AMRs in penman format",
        type=str
    )
    parser.add_argument(
        "-r",
        help="Number of restarts",
        type=int,
        default=10
    )
    # flags
    parser.add_argument(
        "--stop-if-different",
        help="If sentence Smatch is not 1.0, breakpoint",
        action='store_true'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
