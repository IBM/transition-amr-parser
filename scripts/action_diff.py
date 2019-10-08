from collections import Counter
from state_machine import Transitions
import sys


if __name__ == '__main__':

    tokens1 = []
    tokens2 = []

    actions1 = []
    actions2 = []

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    confusion_matrix = {}

    with open(gold_file, 'r') as f:
        sentences = f.read()
    sentences = sentences.replace('\r', '')
    sentences = sentences.split('\n\n')
    for sent in sentences:
        s = sent.split('\n')
        if len(s) < 2:
            continue
        tokens1.append(s[0].split('\t'))
        actions1.append(s[1].split('\t'))

    with open(pred_file, 'r') as f:
        sentences = f.read()
    sentences = sentences.replace('\r', '')
    sentences = sentences.split('\n\n')
    for sent in sentences:
        s = sent.split('\n')
        if len(s) < 2:
            continue
        tokens2.append(s[0].split('\t'))
        actions2.append(s[1].split('\t'))

    for j, _ in enumerate(actions1):
        for i in range(max(len(actions1[j]), len(actions2[j]))):
            a1 = actions1[j][i] if i < len(actions1[j]) else 'pad'
            a2 = actions2[j][i] if i < len(actions2[j]) else 'pad'

            act1 = Transitions.readAction(a1)[0]
            act2 = Transitions.readAction(a2)[0]
            if act1 not in confusion_matrix:
                confusion_matrix[act1] = Counter()
            confusion_matrix[act1][act2] += 1
            if act1 != act2:
                break

    print('gold/pred', end='')
    for act in confusion_matrix:
        print('\t' + act, end='')
    print()

    for act in confusion_matrix:
        print(act, end='\t')
        total = sum(confusion_matrix[act].values())
        for act2 in confusion_matrix:
            print(f'{100*confusion_matrix[act][act2]/total:.1f}', end='\t')
        print()

    print()
    total = sum(confusion_matrix[act][act2] for act in confusion_matrix for act2 in confusion_matrix[act])
    for act in confusion_matrix:
        count = sum(confusion_matrix[act].values())
        print(act, f'{100*count/total:.1f}', end='\t')
