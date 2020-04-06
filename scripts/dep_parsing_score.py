import argparse

from tqdm import tqdm

from transition_amr_parser.io import read_tokenized_sentences


def argument_parser():

    parser = argparse.ArgumentParser(description='Tool to handle AMR')
    parser.add_argument(
        "--in-tokens",
        help="tab separated tokens one sentence per line",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--in-actions",
        help="space separated actions one sentence per line",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--in-gold-actions",
        help="space separated actions one sentence per line",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--out-score",
        help="ARC scores",
        type=str,
    )
    args = parser.parse_args()

    return args


def compute_correct(src_str, hypo_str, target_str):

    def clean_tag(action):
        if 'SHIFT' in action:
            # remove multi-task in SHIFT
            action = action.split('(')[0]
        # for action ngrams, keep only first action
        return action.split(',')[0]

    hypo_str = " ".join([clean_tag(x) for x in hypo_str.split()])
    target_str = " ".join([clean_tag(x) for x in target_str.split()])

    correct_heads = 0
    total_heads = 0
    correct_labels = 0
    gold_heads = {}
    gold_labels = {}
    hyp_heads = {}
    hyp_labels = {}
    pbuffer = []
    pstack = []
    for i in reversed(range(len(src_str.split()))):
        if i<len(src_str.split())-1:
            pbuffer.append(i+1)
        else:
            pbuffer.append(0)
    # compute gold
    for word in target_str.split():

        if "unk" in word:
            word="RIGHT-ARC(preconj)"
        if "SHIFT" in word:
            pstack.append(pbuffer.pop())
        elif "ARC" in word:
            head=-1
            dep=-1
            s0=pstack.pop()
            s1=pstack.pop()
            if "RIGHT" in word:
                head=s1
                dep=s0
            else:
                head=s0
                dep=s1
            pstack.append(head)
            gold_heads[dep]=head
            label=word.split("(")[1]
            gold_labels[dep]=label
        elif word=="SWAP":
            s0=pstack.pop()
            s1=pstack.pop()
            pbuffer.append(s1)
            pstack.append(s0)
    
    pbuffer = []
    pstack = []
    for i in reversed(range(len(src_str.split()))):
        if i<len(src_str.split())-1:
            pbuffer.append(i+1)
        else:
            pbuffer.append(0)

    for word in hypo_str.split():

        if "SHIFT" in word:
            pstack.append(pbuffer.pop())
        elif "ARC" in word:
            head=-1
            dep=-1
            s0=pstack.pop()
            s1=pstack.pop()
            if "RIGHT" in word:
                head=s1
                dep=s0
            else:
                head=s0
                dep=s1
            pstack.append(head)
            hyp_heads[dep]=head
            label=word.split("(")[1]
            hyp_labels[dep]=label  
        elif word=="SWAP":
            s0=pstack.pop()
            s1=pstack.pop()
            pbuffer.append(s1)
            pstack.append(s0)

    for i in range(len(src_str.split())-1):
        idw=i+1
        total_heads +=1
        if idw in gold_heads and idw in hyp_heads:
            if hyp_heads[idw]==gold_heads[idw]:
                correct_heads+=1
            if hyp_heads[idw]==gold_heads[idw] and hyp_labels[idw]==gold_labels[idw]:
                correct_labels+=1

    return total_heads, correct_heads, correct_labels


if __name__ == '__main__':

    args = argument_parser()
    in_tokens = read_tokenized_sentences(args.in_tokens)
    in_actions = read_tokenized_sentences(args.in_actions, separator='\t')
    in_gold_actions = read_tokenized_sentences(args.in_gold_actions)

    assert len(in_tokens) == len(in_actions)
    assert len(in_gold_actions) == len(in_actions)

    num_sentences = len(in_tokens)
    total_heads = 0
    correct_heads = 0
    correct_labels = 0
    for index in tqdm(range(num_sentences)):

        # Compute correct for this sentence
        import ipdb; ipdb.set_trace(context=30)
        total, correct, labels = compute_correct(
            in_tokens[index],
            in_actions[index],
            in_gold_actions[index]
        )

        # update totals
        total_heads += total
        correct_heads += correct
        correct_labels += labels

    uas = correct_heads/total_heads if total_heads > 0 else 0.0
    las = correct_labels/total_heads if total_heads > 0 else 0.0
    print ('UAS: %2.3f %% LAS: %2.3f %%' % (uas*100, las*100))
