from transition_amr_parser.state_machine import AMRStateMachine
from transition_amr_parser.utils import yellow_font

# from smatch
from amr import AMR
import smatch


def read_amr(file_path):
    # read all lines
    ref_amr_lines = []
    with open(file_path, encoding='utf8') as fid:
        line = AMR.get_amr_line(fid)
        while line:
            ref_amr_lines.append(line)
            line = AMR.get_amr_line(fid)
    return ref_amr_lines


def get_amr(tokens, actions, entity_rules):

    # play state machine to get AMR
    state_machine = AMRStateMachine(tokens, entity_rules=entity_rules)
    for action in actions:
        # FIXME: It is unclear that this will be allways the right option
        # manual exploration of dev yielded 4 cases and it works for the 4
        if action == "<unk>":
            action = f'PRED({state_machine.get_top_of_stack()[0].lower()})'
        state_machine.applyAction(action)

    # sanity check: foce close
    if not state_machine.is_closed:
        alert_str = yellow_font('Machine not closed!')
        print(alert_str)
        state_machine.CLOSE()

    # TODO: Probably waisting ressources here
    amr_str = state_machine.amr.toJAMRString()
    return AMR.get_amr_line(amr_str.split('\n'))


def get_smatch_counts(ref_amr, rec_amr, restart_num, counts=False):
    return score_amr_pair(ref_amr, rec_amr, restart_num)


def get_writer(file_path):
    with open(file_path, 'w') as fid:
        def writer(lines):
            nonlocal fid
            for line in lines:
                fid.write(f'{line}\n')
    return writer


class GoldMiner():
    """
    Computes smatch with respect to a reference
    """

    def __init__(self, ref_amr_path, out_actions_path, entity_rules=None):

        # set entity_rules to be used
        self.entity_rules = entity_rules
        # read reference sentences 
        self.ref_amr_lines = read_amr(ref_amr_path)
        self.oracle_smatch_counts_cache = {}
        self.restart_num = 10
        self.mined_actions = {}
        self.original_actions = {}
        self.saved_actions = set()
        # start new file 
        self.out_actions_path = out_actions_path
        self.fid = open(self.out_actions_path, 'w')

    def close(self):
        self.fid.close()

    def update(self, sample_id, tokens, oracle_actions, actions):

        # compute smatch for the rule oracle or retrieve from cache
        if sample_id not in self.oracle_smatch_counts_cache:

            # compute smatch
            oracle_smatch_counts = get_smatch_counts(
                self.ref_amr_lines[sample_id],
                get_amr(tokens, oracle_actions, self.entity_rules),
                self.restart_num
            )
            oracle_smatch = smatch.compute_f(*oracle_smatch_counts)[2]

            # store actions
            self.original_actions[sample_id] = (oracle_actions, oracle_smatch_counts)

            # cache
            self.oracle_smatch_counts_cache[sample_id] = oracle_smatch_counts

        else: 

            # If we outperform oracle keep the amr
            oracle_smatch_counts = self.oracle_smatch_counts_cache[sample_id]
            oracle_smatch = smatch.compute_f(*oracle_smatch_counts)[2]

        # compute smatch for hypothesis
        hypo_counts = get_smatch_counts(
            self.ref_amr_lines[sample_id],
            get_amr(tokens, actions),
            self.restart_num
        )
        hypo_smatch = smatch.compute_f(*hypo_counts)[2]

        # if hypothesis outperforms oracle, keep it
        if oracle_smatch < hypo_smatch:
            actions = [a for a in actions if a != '</s>']
            self.mined_actions[sample_id] = (actions, hypo_counts)

        return hypo_smatch, oracle_smatch

    def save_actions(self):
        """
        Save all what is collected until now
        """

        # index that are still not saved
        indices_seen = set(self.original_actions.keys())
        indices_to_dump = sorted(set(indices_seen) - self.saved_actions)

        # save new indices, if the id is in the mined actions, save that instead
        for did in indices_to_dump:
            # Skip saved actions
            if did in self.saved_actions:
                continue
            if did in self.mined_actions:
                self.fid.write(f'{did} ' + '\t'.join(self.mined_actions[did][0]) + '\n')
            else:
                self.fid.write(f'{did} ' + '\t'.join(self.original_actions[did][0]) + '\n')
            # note this sentence index down
            self.saved_actions.add(did)

        # save progress
        self.fid.close()
        self.fid = open(self.out_actions_path, 'a+')

        # print progress
        self.print_evaluation()

    def print_evaluation(self):

        # compute overall smatch until this point
        original_counts = []
        new_counts = []
        for did in self.original_actions.keys():
            original_counts.append(self.original_actions[did][1])
            if did in self.mined_actions:
                new_counts.append(self.mined_actions[did][1])
            else:
                new_counts.append(self.original_actions[did][1])

        # compute smatch 
        original_smatch = smatch_from_counts(original_counts)
        new_smatch = smatch_from_counts(new_counts)

        print('{:1.3f}'.format(original_smatch))
        print('{:1.3f}'.format(new_smatch))


def smatch_from_counts(counts):
    total_counts = [
        sum([x[0] for x in counts]),
        sum([x[1] for x in counts]),
        sum([x[2] for x in counts])
    ]
    return smatch.compute_f(*total_counts)[2]


def get_smatch_reward(ref_amr_path, task, restart_num=10):
    """
    Closure returning a function that computes smatch given sample batch and
    corresponding hypothesis

    It also computes oracle score for reference, which is cached for speed
    """

    # read entire data first
    ref_amr_lines = read_amr(ref_amr_path)

    # cache for smatch of the rule-based oracle (which is fixed)
    oracle_smatch_counts_cache = {}

    def smatch_reward(sample, hypos):

        nonlocal oracle_smatch_counts_cache
        nonlocal task
        nonlocal restart_num

        # Loop over batch sentences:
        hypo_smatch_counts = []
        for i, sample_id in enumerate(sample['id'].tolist()):

            src_tokens = sample['net_input']['src_tokens'][i, :]

            # compute smatch for the rule oracle or retrieve from cache
            if sample_id not in oracle_smatch_counts_cache:


                # compute smatch
                oracle_smatch_counts_cache[sample_id] = get_smatch_counts(
                    ref_amr_lines[sample_id],
                    get_amr(src_tokens, sample['target'][i, :], task),
                    restart_num
                )

            # compute smatch for hypothesis
            # get amr
            hypo_smatch_counts.append([
                get_smatch_counts(
                    ref_amr_lines[sample_id],
                    get_amr(src_tokens, None, task, hypo=hypo),
                    restart_num
                )
                for hypo in hypos[i]
            ])

            if oracle_smatch_counts_cache[sample_id] < hypo_smatch_counts[-1][0]:
                import ipdb; ipdb.set_trace(context=30)
                print()

        oracle_smatch = [
            oracle_smatch_counts_cache[sample_id]
            for sample_id in sample['id'].tolist()
        ]

        return hypo_smatch_counts, oracle_smatch

    return smatch_reward 


def score_amr_pair(ref_amr_line, rec_amr_line, restart_num, justinstance=False, 
                   justattribute=False, justrelation=False):

    # parse lines
    amr1 = AMR.parse_AMR_line(ref_amr_line)
    amr2 = AMR.parse_AMR_line(rec_amr_line)

    if amr2 is None:
        return 0, 0, len(amr1.get_triples()[0])

    # Fix prefix
    prefix1 = "a"
    prefix2 = "b"
    # Rename node to "a1", "a2", .etc
    amr1.rename_node(prefix1)
    # Renaming node to "b1", "b2", .etc
    amr2.rename_node(prefix2)

    # get triples
    (instance1, attributes1, relation1) = amr1.get_triples()
    (instance2, attributes2, relation2) = amr2.get_triples()

    # optionally turn off some of the node comparison
    doinstance = doattribute = dorelation = True
    if justinstance:
        doattribute = dorelation = False
    if justattribute:
        doinstance = dorelation = False
    if justrelation:
        doinstance = doattribute = False

    (best_mapping, best_match_num) = smatch.get_best_match(
        instance1, attributes1, relation1,
        instance2, attributes2, relation2,
        prefix1, prefix2, 
        restart_num,
        doinstance=doinstance,
        doattribute=doattribute,
        dorelation=dorelation
    )

    if justinstance:
        test_triple_num = len(instance1)
        gold_triple_num = len(instance2)
    elif justattribute:
        test_triple_num = len(attributes1)
        gold_triple_num = len(attributes2)
    elif justrelation:
        test_triple_num = len(relation1)
        gold_triple_num = len(relation2)
    else:
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
    return best_match_num, test_triple_num, gold_triple_num
