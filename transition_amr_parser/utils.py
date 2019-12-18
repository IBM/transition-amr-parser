import itertools
import subprocess
import random
from collections import Counter
from datetime import datetime
from spacy.tokens.doc import Doc
import numpy as np
import torch.nn as nn
import torch.nn.init
from torch.utils.data import Dataset
from torch.nn import functional as F


class AMRDataset(Dataset):

    def __init__(self, sentence_idxs, labelO_tensor, labelA_tensor, action_tensor, pred_tensor):

        self.sentence_idxs = sentence_idxs
        self.labelO_tensor = pad_batch(labelO_tensor) if isinstance(labelO_tensor, list) else labelO_tensor
        self.labelA_tensor = pad_batch(labelA_tensor) if isinstance(labelA_tensor, list) else labelA_tensor
        self.action_tensor = pad_batch(action_tensor) if isinstance(action_tensor, list) else action_tensor
        self.pred_tensor = pad_batch(pred_tensor) if isinstance(pred_tensor, list) else pred_tensor

    def __getitem__(self, index):
        return {'sent_idx': self.sentence_idxs[index],
                'labelsA': self.labelA_tensor[index],
                'labels': self.labelO_tensor[index],
                'actions': self.action_tensor[index],
                'preds': self.pred_tensor[index]}

    def __len__(self):
        return len(self.sentence_idxs)


zip = getattr(itertools, 'izip', zip)


def to_scalar(var):

    return var.view(-1).data.tolist()[0]


def argmax(vec):

    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def vectorize_words(model, words, training=True, random_replace=None, gpu=False):

    if random_replace:
        words = vectorize_random_replace([words], model.word2idx, model.word2idx['<unk>'], random_replace, model.singletons)
    elif training:
        words = vectorize([words], model.word2idx)
    else:
        words = vectorize_safe([words], model.word2idx, model.word2idx['<unk>'])
    return torch.LongTensor(words).cuda() if gpu else torch.LongTensor(words)


def vectorize(input_lines, word_dict):
    lines = [[word_dict[w] for w in line] for line in input_lines]
    return lines


def vectorize_safe(input_lines, word_dict, unk):
    lines = [[word_dict.get(w, unk) for w in line] for line in input_lines]
    return lines


def vectorize_random_replace(input_lines, word_dict, unk, rate, singletons):
    lines = [[word_dict[w] for w in line] for line in input_lines]
    for i, sent in enumerate(lines):
        for j, word in enumerate(sent):
            if word in singletons:
                r = random.random()
                if r < rate:
                    lines[i][j] = unk
    return lines


def construct_dataset_train(model, correct_transitions, gpu=False):

    input_idxs = list()
    input_words = list()
    input_labelO = list()
    input_labelA = list()
    input_action = list()
    input_pred = list()

    for i, tr in enumerate(correct_transitions):
        copy = tr.amr.tokens.copy()
        input_idxs.append(i)
        input_words.append(copy)
        actions = [tr.readAction(a)[0] for a in tr.actions]

        input_action.append(actions)
        input_labelO.append(tr.labels)
        input_labelA.append(tr.labelsA)
        input_pred.append(tr.predicates)

    labelsO = vectorize(input_labelO, model.labelsO2idx)
    labelsA = vectorize(input_labelA, model.labelsA2idx)
    actions = vectorize(input_action, model.action2idx)
    preds = vectorize(input_pred, model.pred2idx)

    dataset = AMRDataset(input_idxs,
                         [torch.LongTensor(l).cuda() if gpu else torch.LongTensor(l) for l in labelsO],
                         [torch.LongTensor(l).cuda() if gpu else torch.LongTensor(l) for l in labelsA],
                         [torch.LongTensor(a).cuda() if gpu else torch.LongTensor(a) for a in actions],
                         [torch.LongTensor(p).cuda() if gpu else torch.LongTensor(p) for p in preds])

    return dataset, input_words


def xavier_init(gpu, *size):
    t = torch.FloatTensor(*size)
    if gpu:
        t = t.cuda()
    return nn.init.xavier_normal_(t)


def initialize_embedding(input_embedding):

    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def initialize_zero(input_linear):
    nn.init.zeros_(input_linear.weight)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def initialize_linear(input_nn, orth=True):

    if orth:
        nn.init.orthogonal_(input_nn.weight)
    else:
        nn.init.xavier_normal_(input_nn.weight)

    if input_nn.bias is not None:
        nn.init.zeros_(input_nn.bias)


def initialize_lstm(in_lstm):

    for ind in range(0, in_lstm.num_layers):
        weight = eval('in_lstm.weight_ih_l'+str(ind))
        nn.init.orthogonal_(weight)
        weight = eval('in_lstm.weight_hh_l'+str(ind))
        nn.init.orthogonal_(weight)

    if in_lstm.bias:
        start, end = in_lstm.hidden_size, 2 * in_lstm.hidden_size
        for ind in range(0, in_lstm.num_layers):
            bias = eval('in_lstm.bias_ih_l'+str(ind))
            nn.init.zeros_(bias)
            bias.data[start: end] = 1
            bias = eval('in_lstm.bias_hh_l'+str(ind))
            nn.init.zeros_(bias)
            bias.data[start: end] = 1


def initialize_lstm_cell(in_lstm_cell):

    weight = eval('in_lstm_cell.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)
    weight = eval('in_lstm_cell.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)

    if in_lstm_cell.bias:
        start, end = in_lstm_cell.hidden_size, 2 * in_lstm_cell.hidden_size
        bias = eval('in_lstm_cell.bias_ih')
        nn.init.zeros_(bias)
        bias.data[start: end] = 1
        bias = eval('in_lstm_cell.bias_hh')
        nn.init.zeros_(bias)
        bias.data[start: end] = 1


def make_efficient(use_gpu, labelO, labelA, action, pred):
    if use_gpu:
        labelO = labelO.contiguous().cuda()
        labelA = labelA.contiguous().cuda()
        action = action.contiguous().cuda()
        pred = pred.contiguous().cuda()
    else:
        labelO = labelO.contiguous()
        labelA = labelA.contiguous()
        action = action.contiguous()
        pred = pred.contiguous()
    return labelO, labelA, action, pred


def reverse_sequence(seq, dim=0, gpu=False):
    reversed_idx = torch.LongTensor([i for i in reversed(range(seq.size(dim)))])
    if gpu:
        reversed_idx = reversed_idx.cuda()
    return seq.index_select(dim, reversed_idx)


def pad_batch(batch_list):
    max_len = max(sent.size()[-1] for sent in batch_list)
    for i, ex in enumerate(batch_list):
        pad_size = max_len-ex.size()[-1]
        if pad_size > 0:
            batch_list[i] = F.pad(ex, pad=[0, pad_size])
        if len(batch_list[i].size()) == 1:
            batch_list[i] = batch_list[i].unsqueeze(0)
    batch = torch.cat(batch_list, 0)
    return batch


def pad_batch_tokens(batch_list):
    max_len = max(len(sent) for sent in batch_list)
    for i, ex in enumerate(batch_list):
        pad_size = max_len-len(ex)
        if(pad_size > 0):
            for j in range(pad_size):
                batch_list[i].append('<eof>')
    batch = batch_list
    return batch


def set_seed(seed):
    random.seed(seed)


class Accuracy:

    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, bool):
        if bool:
            self.correct += 1
        self.total += 1

    def val(self):
        # prec = self.true_positive/self.positive if self.positive else 0
        # rec = self.true_positive/self.true if self.true else 0
        # return 2*(prec*rec)/(prec+rec)
        return self.correct / self.total if self.total > 0 else 0

    def __str__(self):
        return f'{self.val():.2f}'

    def reset(self):
        self.total = 0
        self.correct = 0

    def data_as_tensor(self):
        data = [self.total, self.correct]
        t = torch.from_numpy(np.array(data))
        return t

    def reset_from_tensor(self, tensor):
        self.reset()
        self.total = tensor[0].item()
        self.correct = tensor[1].item()


class ConfusionMatrix:

    def __init__(self, labels):
        self.labels = list(sorted(l for l in labels))
        self.confusion_matrix = {l: Counter() for l in self.labels}

    def reset(self):
        self.confusion_matrix = {l: Counter() for l in self.labels}

    def add(self, label1, label2):
        self.confusion_matrix[label1][label2] += 1

    def data_as_tensor(self):
        data = [[self.confusion_matrix[label1][label2] for label2 in self.labels] for label1 in self.labels]
        t = torch.from_numpy(np.array(data))
        return t

    def reset_from_tensor(self, tensor):
        self.reset()
        for i, l1 in enumerate(self.labels):
            for j, l2 in enumerate(self.labels):
                self.confusion_matrix[l1][l2] = tensor[i, j].item()

    def __str__(self):

        totals = Counter({act: sum(self.confusion_matrix[act].values()) for act in self.labels})
        totals_all = sum(totals.values())
        labels = sorted(self.labels, key=lambda x: totals[x], reverse=True)

        s = '|gold/pred %|'
        for act in labels:
            s += act + '|'
        s += '\n'
        s += '|-|'
        for _ in labels:
            s += '-|'
        s += '\n'
        for act in labels:
            if act == '<pad>':
                continue
            s += f'|{act} {100 * totals[act] / totals_all:.1f}|'
            for act2 in labels:
                if act == act2:
                    s += f'**{100 * self.confusion_matrix[act][act2] / totals[act] if totals[act] else float("nan"):.1f}**|'
                else:
                    s += f'{100 * self.confusion_matrix[act][act2] / totals[act] if totals[act] else float("nan"):.1f}|'
            s += '\n'
        return s


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


def print_log(module, string):
    """formats printing of log to stdout"""
    timestamp = str(datetime.now()).split('.')[0]
    print(f'{timestamp} [{module}] {string}')


def smatch_wrapper(amr_dev_data, predicted_amr_file, significant=3):
    """Calls smatch.py on the command line and parses output"""

    # Command line call
    cmd = f'python smatch/smatch.py --significant {significant} -f {amr_dev_data} {predicted_amr_file}'
    cmd_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # parse output
    smatch_stdout = [line.decode('utf-8') for line in cmd_proc.stdout]
    _, smatch_score = smatch_stdout[0].split()
    return smatch_score


class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)
