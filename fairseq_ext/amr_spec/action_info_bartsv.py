import os
import sys

from tqdm import tqdm

from transition_amr_parser.amr_machine import AMRStateMachine
from fairseq_ext.amr_reform.o10_action_reformer_subtok import AMRActionReformerSubtok


def get_actions_states(*, tokens=None, tokseq_len=None, actions=None, dictionary=None, machine_config=None):
    """Get the information along with runing the AMR state machine in canonical mode with the provided action sequence.

    The information includes:
        - the allowed actions (canonical form) for each action in the actions sequence ('CLOSE' position included)
        - a mask on previous actions to indicate node generation
        - token cursor before each action

    Args:
        tokens (List[str], optional): source token sequences (a sentence); must end with "<ROOT>". Defaults to None.
        tokseq_len (int, optional): source token sequence length. Defaults to None.
        actions (List[str], optional): the action sequence. Defaults to None.

    Returns:
        dict, with keys:
            - allowed_cano_actions (List[List[str]]): allowed canonical actions for each action position. This includes
                the last position for "CLOSE" action, whether it is in the "actions" input or not.
            - actions_nodemask (list): a list of 0 or 1 to indicate which actions generate node
            - token_cursors (list): a list of token cursors before each action is applied

    Note:
        - "tokens" could be None, in which case "tokseq_len" should not be None.
        - when "tokens" is not None, "tokseq_len" is ignored.
    """
    assert actions is not None
    if actions[-1] != 'CLOSE':
        actions = actions.copy()
        actions.append('CLOSE')

    assert tokens is not None
    assert machine_config is not None

    amr_state_machine = AMRStateMachine.from_config(machine_config)

    actions_states = AMRActionReformerSubtok.reform_actions_and_get_states(tokens, actions,
                                                                           dictionary, amr_state_machine,
                                                                           restrict_subtoken=True)

    return actions_states


def check_actions_file(en_file, actions_file, dictionary, machine_config, out_file=None):
    """Run the AMR state machine in canonical mode for pairs of English sentences and actions, to check the validity
    of the rules for allowed actions, and output data statistics.

    Args:
        en_file (str): English sentence file path.
        actions_file (str): actions file path.

    Returns:

    """
    avg_num_allowed_actions_pos = 0
    avg_num_allowed_actions_seq = 0
    avg_num_arcs_pos = 0
    avg_num_arcs_not1st_seq = 0
    num_pos = 0
    num_seq = 0
    avg_len_en = 0
    avg_len_actions = 0
    with open(en_file, 'r') as f, open(actions_file, 'r') as g:
        for tokens, actions in tqdm(zip(f, g)):
            if tokens.strip():
                tokens = tokens.strip().split('\t')
                actions = actions.strip().split('\t')
                # assert tokens[-1] == '<ROOT>'
                actions_states = get_actions_states(tokens=tokens, actions=actions,
                                                    dictionary=dictionary, machine_config=machine_config)
                # breakpoint()
                # get statistics
                allowed_cano_actions = actions_states['allowed_cano_actions']
                num_pos += len(allowed_cano_actions)    # this includes the last "CLOSE" action
                num_seq += 1
                avg_len_en += len(tokens)
                # avg_len_actions += len(actions)
                avg_len_actions += len(actions_states['actions_nopos_out'])
                avg_num_allowed_actions_pos += sum(map(len, allowed_cano_actions))
                avg_num_allowed_actions_seq += len(list(set.union(*map(set, allowed_cano_actions))))
                actions_cano = map(AMRStateMachine.from_config(machine_config).get_base_action, actions)
                avg_num_arcs_pos += len(list(filter(lambda act: act.startswith('>LA') or act.startswith('>RA'),
                                                    actions_cano)))
                avg_num_arcs_not1st_seq += len([1 for a, b in zip(actions, actions[1:])
                                                if (a.startswith('>LA') or a.startswith('>RA'))
                                                and (b.startswith('>LA') or b.startswith('>RA'))])

    avg_num_allowed_actions_pos /= num_pos
    avg_num_allowed_actions_seq /= num_seq
    avg_num_arcs_pos /= num_pos
    avg_num_arcs_not1st_seq /= num_seq
    avg_len_en /= num_seq
    avg_len_actions /= num_seq

    print(
        f'number of sequences: {num_seq}, number of action tokens (including CLOSE): {num_pos}',
        file=out_file or sys.stdout)
    print(
        f'average en sentence length (including <ROOT>): {avg_len_en}, '
        f'average actions sequence length (excluding CLOSE): {avg_len_actions}',
        file=out_file or sys.stdout)
    print(
        f'average number of arc actions per action token position (excluding CLOSE): {avg_num_arcs_pos}',
        file=out_file or sys.stdout)
    print(
        f'average number of arc actions that are not the 1st arc action inside an arc subsequence per action sequence: '
        f'{avg_num_arcs_not1st_seq}',
        file=out_file or sys.stdout)
    print(
        f'average number of allowed canonical actions per action token position: {avg_num_allowed_actions_pos}',
        file=out_file or sys.stdout)
    print(
        f'average number of allowed canonical actions per action sequence: {avg_num_allowed_actions_seq} (max {7})',
        file=out_file or sys.stdout)


if __name__ == '__main__':
    for split in ['train', 'dev', 'test']:
        print('-' * 20)
        print(split + ' data')
        print('-' * 20)

        en_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.en'
        actions_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.actions'
        out_file_path = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/EXP/exp1/oracle/{split}.stats'

        en_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/test_data/oracles/o7+Word100/{split}.en'
        actions_file = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/test_data/oracles/o7+Word100/{split}.actions'
        out_file_path = f'/dccstor/ykt-parse/AMR/jiawei2020/transition-amr-parser/test_data/oracles/o7+Word100/{split}.stats'

        en_file = f'/dccstor/jzhou1/work/EXP/data/o3-prefix_act-states/oracle/{split}.en'
        actions_file = f'/dccstor/jzhou1/work/EXP/data/o3-prefix_act-states/oracle/{split}.actions'
        out_file_path = f'/dccstor/jzhou1/work/EXP/data/o3-prefix_act-states/oracle/{split}.stats'

        en_file = f'/dccstor/jzhou1/work/EXP/data-amr1/depfix_o5_no-mw_act-states/oracle/{split}.en'
        actions_file = f'/dccstor/jzhou1/work/EXP/data-amr1/depfix_o5_no-mw_act-states/oracle/{split}.actions'
        out_file_path = f'/dccstor/jzhou1/work/EXP/data-amr1/depfix_o5_no-mw_act-states/oracle/{split}.stats'

        en_file = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser-o8/EXP/data/o8.3_act-states/oracle/{split}.en'
        actions_file = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser-o8/EXP/data/o8.3_act-states/oracle/{split}.actions'
        out_file_path = f'/cephfs_nese/TRANSFER/rjsingh/DDoS/DDoS/jzhou/transition-amr-parser-o8/EXP/data/o8.3_act-states/oracle/{split}.stats'

        en_file = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/{split}.tokens'
        actions_file = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/{split}.actions'
        machine_config = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/machine_config.json'
        out_file_path = f'/n/tata_ddos_ceph/jzhou/transition-amr-parser-bart-o10/DATA/AMR2.0/oracles/o10/{split}.stats'
        from fairseq_ext.data.amr_bpe import AMRActionBartDictionary
        node_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.nodes'
        others_file_path = 'DATA/AMR2.0/oracles/o10/train.actions.vocab.others'
        dictionary = AMRActionBartDictionary(node_freq_min=5,
                                             node_file_path=node_file_path,
                                             others_file_path=others_file_path)

        out_file = open(out_file_path, 'w')
        check_actions_file(en_file, actions_file, dictionary, machine_config, out_file)
        out_file.close()
        os.system(f'cat {out_file_path}')
        print('\n' + f'stats saved to {out_file_path}')
        break
