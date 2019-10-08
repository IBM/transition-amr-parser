import socket
import time
from datetime import datetime, timedelta
import warnings
from collections import Counter
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from amr import JAMR_CorpusReader
from state_machine import Transitions
import stack_lstm as sl
import utils
from utils import print_log, smatch_wrapper
from data_oracle import AMR_Oracle

from tqdm import tqdm
import h5py
import spacy
from spacy.tokens.doc import Doc

import random

warnings.simplefilter(action='ignore', category=FutureWarning)


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser.')
    parser.add_argument("-A", "--amr_training_data", required=True)
    parser.add_argument("-a", "--amr_dev_data", required=True)
    parser.add_argument('--epoch', type=int, default=300, help='maximum epoch number')
    parser.add_argument('--report', type=int, default=1, help='after how many epochs should the model evaluate on dev data?')
    parser.add_argument('--max_train_size', type=int, help='number of sentences to train on (for debugging purposes)')
    parser.add_argument('--max_dev_size', type=int, help='number of sentences to evaluate on (for debugging purposes)')
    parser.add_argument('--name', help='name of experiment to associate with all output (default: process id)')
    parser.add_argument('--desc', help='description of experiment; to be printed at the top of smatch file')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    parser.add_argument('--write_actions', action='store_true', help='whether to output predicted actions')
    parser.add_argument('--write_gold_actions', help='file to output gold actions')
    parser.add_argument('--read_gold_actions', help='use gold actions from this file')
    parser.add_argument('--load_model', help='use parameters file to initialize modal')
    parser.add_argument('--confusion', action='store_true', help='write confusion matrix')
    parser.add_argument('--gpu', action='store_true', help='use GPU processor')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--test_mode', action='store_true', help='do not train')
    parser.add_argument('--save_model', help='store model and metrics here')
    # hyperparams
    parser.add_argument('--dim_emb', type=int, default=100, help='embedding dim')
    parser.add_argument('--dim_action', type=int, default=100, help='action embedding dim')
    parser.add_argument('--dim_char_emb', type=int, default=50, help='char embedding dim')
    parser.add_argument('--dim_hidden', type=int, default=100, help='hidden dim')
    parser.add_argument('--dim_char_hidden', type=int, default=50, help='char hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='decay ratio of learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0005, help='minimum learning rate to be reach')
    parser.add_argument('--patience', type=int, default=0, help='number of epochs to wait before reducing learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--replace_unk', type=float, default=0.1, help='rate to replace training tokens with unk')
    # ablations
    parser.add_argument('--no_chars', action='store_true', help='do not use character embeddings as input')
    parser.add_argument('--no_attention', action='store_true', help='do not use attention over input sentence')
    parser.add_argument('--no_bert', action='store_true', help='do not use bert embeddings')
    # experiments
    parser.add_argument('--obj', default='ML', help='set RL or ML objective, default is ML')
    parser.add_argument('--adam', action='store_true', help='use Adam instead of SGD')
    parser.add_argument('--adadelta', action='store_true', help='use Adadelta instead of SGD')
    parser.add_argument('--function_words', action='store_true', help='use function words')
    parser.add_argument('--function_words_rels', action='store_true', help='use function words (relations only)')
    parser.add_argument('--parse_unaligned', action='store_true', help='parse unaligned nodes')
    parser.add_argument('--weight_inputs', action='store_true', help='softmax weight inputs')
    parser.add_argument('--attend_inputs', action='store_true', help='attention weight inputs as a function of the stack')
    # pretrained bert embeddings
    parser.add_argument('--pretrained_dim', type=int, default=1024, help='pretrained bert dim')
    parser.add_argument("-B", "--bert_training", required=False)
    parser.add_argument("-b", "--bert_test", required=False)
    # tests
    parser.add_argument('--unit_tests', action='store_true', help='test parser')
    # multiprocess
    parser.add_argument('--cores', type=int, default=10, help='number of cores')
    parser.add_argument('--batch', type=int, default=1, help='number of sentences per batch')

    args = parser.parse_args()

    if args.unit_tests:
        print('[run tests] Testing parser with default parameters and a small dataset', file=sys.stderr)
        for _ in range(5):
            print('>', file=sys.stderr)
        args.seed = 0
        args.max_train_size = 100
        args.max_dev_size = 100
        args.epoch = 1
        args.name = 'unit_tests'

    if args.test_mode:
        args.epoch = 1
        args.max_train_size = 0

    return args


def main():

    # store time of start
    start_timestamp = str(datetime.now()).split('.')[0]

    # Argument handling
    args = argument_parser()

    # Initialization
    if args.seed is not None:
        torch.manual_seed(args.seed)
        utils.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
    exp_name = args.name if args.name else f'{os.getpid()}'

    # Oracle computation
    cr = JAMR_CorpusReader()
    cr.load_amrs(args.amr_training_data, training=True)
    cr.load_amrs(args.amr_dev_data, training=False)

    oracle = AMR_Oracle(verbose=False)

    add_unaligned = 10 if args.parse_unaligned else 0
    if args.read_gold_actions:
        oracle.transitions = oracle.read_actions(args.read_gold_actions)
    else:
        oracle.runOracle(cr.amrs, add_unaligned=add_unaligned, action_file=args.write_gold_actions)

    train_amrs = oracle.gold_amrs
    dev_sentences = [
        amr.tokens + (['<unaligned>'] * add_unaligned + ['<ROOT>'])
        for amr in cr.amrs_dev]

    if args.max_dev_size is not None:
        dev_sentences = dev_sentences[:args.max_dev_size]

    # BERT embeddings
    use_bert = not args.no_bert
    h5py_train, h5py_test = None, None
    if use_bert:
        h5py_train, h5py_test = setup_bert(args.bert_training, args.bert_test)

    model = AMRModel(amrs=train_amrs,
                     oracle_transitions=oracle.transitions,
                     possible_predicates=oracle.possiblePredicates,
                     embedding_dim=args.dim_emb,
                     action_embedding_dim=args.dim_action,
                     char_embedding_dim=args.dim_char_emb,
                     hidden_dim=args.dim_hidden,
                     char_hidden_dim=args.dim_char_hidden,
                     rnn_layers=args.layers,
                     dropout_ratio=args.dropout,
                     pretrained_dim=args.pretrained_dim,
                     use_bert=use_bert,
                     use_gpu=args.gpu,
                     use_chars=not args.no_chars,
                     use_attention=not args.no_attention,
                     use_function_words=args.function_words,
                     use_function_words_rels=args.function_words_rels,
                     parse_unaligned=args.parse_unaligned,
                     weight_inputs=args.weight_inputs,
                     attend_inputs=args.attend_inputs
                     )
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    scheduler = None
    if args.adam:
        optimizer = optim.Adam(model.parameters())
    elif args.adadelta:
        optimizer = optim.Adadelta(model.parameters())
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'max',
            factor=args.lr_decay,
            patience=args.patience,
            min_lr=args.min_lr,
            verbose=True
        )

    data, data_tokens = utils.construct_dataset_train(model, oracle.transitions, gpu=args.gpu)

    # multiprocess
    master_addr, master_port = setup_multiprocess(args)
    if use_bert:
        print_log('parser', 'Loading BERT')
        bert_embeddings_train = [get_bert_embeddings(h5py_train, id, tok) for id, tok in enumerate(data_tokens)]
    else:
        bert_embeddings_train = None

    tot_length = len(data)
    if args.max_train_size:
        tot_length = min(args.max_train_size, tot_length)

    # Start log of epoch by epoch improvements, include timestamp
    if args.save_model:
        smatch_file = f'{args.save_model}/{exp_name}_smatch.txt'
    else:
        smatch_file = f'{exp_name}_smatch.txt'
    with open(smatch_file, 'w+') as f:
        desc = args.desc if args.desc else ''
        f.write(f'{start_timestamp}\t{desc}\n')

    training = not args.test_mode
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    print_log('parser', 'Start Parsing')
    for epoch_idx, _ in enumerate(epoch_list):
        start = time.time()

        if training:
            if epoch_idx < 10:
                model.warm_up = True
            arguments = (
                model, args, data, data_tokens, bert_embeddings_train,
                optimizer, master_addr, master_port, args.cores,
                exp_name, epoch_idx
            )
            if args.cores > 1:
                mp.spawn(train_worker, nprocs=args.cores, args=arguments)
            else:
                # call code for single thread, add dummy rank 0
                train_worker(*(tuple([0]) + arguments))
            model.warm_up = False

        end = time.time()
        print_log('parser', f'{timedelta(seconds=int(end-start))}')
        # create output
        if (epoch_idx + 1 - args.start_epoch) % args.report == 0:

            # save model
            if args.save_model:
                parameters_file = f'{args.save_model}/{exp_name}.epoch{epoch_idx}.params'
            else:
                parameters_file = f'{exp_name}.epoch{epoch_idx}.params'
            print_log('parser', f'Saving model to: {parameters_file}')
            torch.save(model.state_dict(), parameters_file)

            start = time.time()
            smatch_score = eval_parser(exp_name, model, args, dev_sentences,
                                       h5py_test, epoch_idx, smatch_file,
                                       args.save_model, optimizer.param_groups)

            end = time.time()
            print_log('evaluation', f'{timedelta(seconds=int(end-start))}')

            if scheduler is not None and smatch_score is not None:
                scheduler.step(smatch_score)

    print_log('parser', 'Done')


def train_worker(rank, model, args, data, data_tokens, bert_embeddings_train,
                 optimizer, master_addr, master_port, cores, exp_name, epoch_idx):
    model.train()
    model.reset_stats()

    print_log('train', f'starting process {rank}')
    if cores > 1:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(world_size=cores, backend='gloo', rank=rank)
        dist_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        # Get only slice of data for this core
        dataset_loader = DataLoader(
            data,
            args.batch,
            drop_last=False,
            sampler=DistributedSampler(data, cores, rank)
        )
    else:
        dist_model = model
        # Get all data
        dataset_loader = DataLoader(data, args.batch, drop_last=False)

    total_len = len(data) // cores
    if args.max_train_size:
        total_len = min(total_len, args.max_train_size // cores)

    # train on sentence
    i = 0
    for batch in tqdm(dataset_loader, desc=f'[train] epoch {epoch_idx}'):
        sent_idx, labelO, labelA, action, pred = batch['sent_idx'], batch['labels'], batch['labelsA'], batch['actions'], batch['preds']

        batch_size = sent_idx.size()[0]
        tokens = utils.pad_batch_tokens([data_tokens[sent_idx[i].item()] for i in range(batch_size)])

        sent_emb = utils.pad_batch(
            [utils.vectorize_words(model, tokens[i], training=True, random_replace=args.replace_unk, gpu=args.gpu) for i in range(batch_size)])
        bert_emb = [bert_embeddings_train[sent_idx[i].item()] for i in range(batch_size)] if not args.no_bert else None

        labelO, labelA, action, pred = utils.make_efficient(args.gpu, labelO, labelA, action, pred)

        dist_model.zero_grad()

        loss = dist_model.forward(sent_idx, sent_emb, labelO, labelA, action, pred, args.obj, tokens=tokens, bert_embedding=bert_emb)

        loss.backward()

        model.epoch_loss += loss.sum().item()
        nn.utils.clip_grad_norm_(dist_model.parameters(), args.clip_grad)
        optimizer.step()

        i += args.batch
        if args.max_train_size and i > total_len:
            break

    # consolidate stats between different processes
    # (This is a trick to send info using pytorch's
    # interface for sending tensors between multiple processes)
    losses = torch.FloatTensor([model.epoch_loss, model.action_loss,
                                model.label_loss, model.labelA_loss,
                                model.pred_loss])

    # Get scores
    action_acc = model.action_acc.data_as_tensor()
    label_acc = model.label_acc.data_as_tensor()
    labelA_acc = model.labelA_acc.data_as_tensor()
    pred_acc = model.pred_acc.data_as_tensor()
    if args.confusion:
        act_conf = model.action_confusion_matrix.data_as_tensor()
        label_conf = model.label_confusion_matrix.data_as_tensor()

    if cores > 1:
        # Sync data across threads
        dist.reduce(losses, dst=0)
        dist.reduce(action_acc, dst=0)
        dist.reduce(label_acc, dst=0)
        dist.reduce(labelA_acc, dst=0)
        dist.reduce(pred_acc, dst=0)
        if args.confusion:
            dist.reduce(act_conf, dst=0)
            dist.reduce(label_conf, dst=0)

    if rank == 0:
        model.epoch_loss, model.action_loss, model.label_loss, model.labelA_loss, model.pred_loss = \
            losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item(), losses[4].item()
        model.action_acc.reset_from_tensor(action_acc)
        model.label_acc.reset_from_tensor(label_acc)
        model.labelA_acc.reset_from_tensor(labelA_acc)
        model.pred_acc.reset_from_tensor(pred_acc)
        if args.confusion:
            model.action_confusion_matrix.reset_from_tensor(act_conf)
            model.label_confusion_matrix.reset_from_tensor(label_conf)

        # Inform user
        print_epoch_report(model, exp_name, epoch_idx, args.weight_inputs, args.attend_inputs, args.parse_unaligned, args.confusion, args.save_model)


def eval_parser(exp_name, model, args, dev_sentences, h5py_test, epoch_idx, smatch_file, save_model, param_groups):

    # save also current learning rate
    learning_rate = ",".join([str(x['lr']) for x in param_groups])

    model.eval()

    # evaluate on dev
    print_log('eval', f'Evaluating on: {args.amr_dev_data}')
    if save_model:
        predicted_amr_file = f'{save_model}/{exp_name}_amrs.epoch{epoch_idx}.dev.txt'
    else:
        predicted_amr_file = f'{exp_name}_amrs.epoch{epoch_idx}.dev.txt'
    with open(predicted_amr_file, 'w+') as f:
        f.write('')
    print_log('eval', f'Writing amr graphs to: {predicted_amr_file}')
    if save_model:
        actions_file = f'{save_model}/{exp_name}_actions.epoch{epoch_idx}.dev.txt'
    else:
        actions_file = f'{exp_name}_actions.epoch{epoch_idx}.dev.txt'
    if args.write_actions:
        print_log('eval', f'Writing actions to: {actions_file}')
        with open(actions_file, 'w+') as f:
            f.write('')

    sent_idx = 0
    dev_hash = 0
    for tokens in tqdm(dev_sentences):

        sent_rep = utils.vectorize_words(model, tokens, training=False, gpu=args.gpu)
        dev_b_emb = get_bert_embeddings(h5py_test, sent_idx, tokens) if not args.no_bert else None

        _, actions, labels, labelsA, predicates = model.forward_single(
            sent_rep,
            mode='predict',
            tokens=tokens,
            bert_embedding=dev_b_emb
        )

        # write amr graphs
        apply_actions = []
        for act, label, labelA, predicate in zip(actions, labels, labelsA, predicates):
            # print(act, label, labelA, predicate)
            if act.startswith('PR'):
                apply_actions.append(act + f'({predicate})')
            elif act.startswith('RA') or act.startswith('LA') and not act.endswith('(root)'):
                apply_actions.append(act + f'({label})')
            elif act.startswith('AD'):
                apply_actions.append(act + f'({labelA})')
            else:
                apply_actions.append(act)
        if args.unit_tests:
            dev_hash += sum(model.action2idx[a] for a in actions)
            dev_hash += sum(model.labelsO2idx[l] for l in labels if l)
            dev_hash += sum(model.labelsA2idx[l] for l in labelsA if l)
            dev_hash += sum(model.pred2idx[p] if p in model.pred2idx else 0 for p in predicates if p)

        # print('[eval]',apply_actions)
        if args.write_actions:
            with open(actions_file, 'a') as f:
                f.write('\t'.join(tokens) + '\n')
                f.write('\t'.join(apply_actions) + '\n\n')
        tr = Transitions(tokens, verbose=False)
        tr.applyActions(apply_actions)
        with open(predicted_amr_file, 'a') as f:
            f.write(tr.amr.toJAMRString())
        sent_idx += 1
    # run smatch
    print_log('eval', f'Computing SMATCH')
    smatch_score = smatch_wrapper(
        args.amr_dev_data,
        predicted_amr_file,
        significant=3
    )
    print_log('eval', f'SMATCH: {smatch_score}')
    timestamp = str(datetime.now()).split('.')[0]

    # store all information in file
    print_log('eval', f'Writing SMATCH and other info to: {smatch_file}')
    with open(smatch_file, 'a') as fid:
        fid.write("\t".join([
            f'epoch {epoch_idx}',
            f'learning_rate {learning_rate}',
            f'time {timestamp}',
            f'F-score {smatch_score}\n'
        ]))

    if args.unit_tests:
        test1 = (model.epoch_loss == 3360.1150283813477)
        test2 = (dev_hash == 6038)
        print(f'[run tests] epoch_loss==3360.1150283813477 (got {model.epoch_loss}) {"pass" if test1 else "fail"}',
              file=sys.stderr)
        print(f'[run tests] dev hash==6038 (got {dev_hash}) {"pass" if test2 else "fail"}', file=sys.stderr)
        assert (test1)
        assert (test2)

    return smatch_score


def setup_bert(bert_train_file, bert_test_file):
    if not bert_train_file or not bert_test_file:
        raise Exception('Bert training and test embeddings should be provided (otherwise use --no_bert)')
    print_log('parser', f'BERT train: {bert_train_file}')
    h5py_train = h5py.File(bert_train_file, 'r')
    print_log('parser', f'BERT test: {bert_test_file}')
    h5py_test = h5py.File(bert_test_file, 'r')
    return h5py_train, h5py_test


def get_bert_embeddings(h5py, sent_idx, tokens):
    b_emb = []
    for i, word in enumerate(tokens):
        line = h5py.get(str(sent_idx))
        if line is None:
            raise Exception(f'BERT failed to find embedding: {sent_idx}, {tokens}')
        if word not in ['<ROOT>', '<unaligned>', '<eof>']:
            embedding = line[i]
            b_emb.append(embedding)
    return b_emb


def setup_multiprocess(args):
    if not (dist.is_available()):
        print_log('parser', f'Warning: Distributed processing unavailable. Defaulting to single process.', file=sys.stderr)
        args.cores = 1
        return '', ''
    master_addr = socket.gethostname()
    master_port = '64646'
    print_log('parser', f'multiprocessing with {args.cores} processes')
    print_log('parser', f'multiprocessing ADDR: {master_addr} PORT: {master_port}')
    return master_addr, master_port


def print_epoch_report(model, exp_name, epoch_idx, weight_inputs, attend_inputs, parse_unaligned, confusion, save_model):

    print_log('train', f'Loss {model.epoch_loss}')
    print_log('train', f'actions: {model.action_loss} labels: {model.label_loss} labelsA: {model.labelA_loss} predicates: {model.pred_loss}')
    print_log('train', f'Accuracy actions: {model.action_acc} labels: {model.label_acc} labelsA: {model.labelA_acc} predicates: {model.pred_acc}')

    if weight_inputs:
        print_log('train', f'(Attention) Stack   Buffer  Actions Attention   [other]')
        print_log('train', f'action attention: {F.softmax(model.action_attention, dim=0).tolist()}')
        print_log('train', f'label attention: {F.softmax(model.label_attention, dim=0).tolist()}')
        print_log('train', f'labelA attention: {F.softmax(model.labelA_attention, dim=0).tolist()}')
        print_log('train', f'pred attention: {F.softmax(model.pred_attention, dim=0).tolist()}')
        if parse_unaligned:
            print_log('train', f'pred unaligned attention: {F.softmax(model.pred_attention_unaligned, dim=0).tolist()}')
    if attend_inputs:
        print_log('train', f'(Attention bias) Stack   Buffer  Actions Attention   [other]')
        print_log('train', f'action attention: {F.softmax(model.action_attention.bias, dim=0).tolist()}')
        print_log('train', f'label attention: {F.softmax(model.label_attention.bias, dim=0).tolist()}')
        print_log('train', f'labelA attention: {F.softmax(model.labelA_attention.bias, dim=0).tolist()}')
        print_log('train', f'pred attention: {F.softmax(model.pred_attention.bias, dim=0).tolist()}')
        if parse_unaligned:
            print_log('train', f'pred unaligned attention: {F.softmax(model.pred_attention_unaligned.bias, dim=0).tolist()}')

    if confusion:
        if save_model:
            confusion_file = f'{save_model}/{exp_name}_confusion_matrix.epoch{epoch_idx}.train.txt'
        else:
            confusion_file = f'{exp_name}_confusion_matrix.epoch{epoch_idx}.train.txt'
        print_log('train', f'Writing confusion matrix to: {confusion_file}')
        with open(confusion_file, 'w+') as f:
            f.write('Actions:\n')
            f.write(str(model.action_confusion_matrix) + '\n')
            f.write('Labels:\n')
            f.write(str(model.label_confusion_matrix) + '\n')


class NoTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        spaces = [True] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)


def build_amr(tokens, actions, labels, labelsA, predicates):
    apply_actions = []
    for act, label, labelA, predicate in zip(actions, labels, labelsA, predicates):
        # print(act, label, labelA, predicate)
        if act.startswith('PR'):
            apply_actions.append(act + f'({predicate})')
        elif act.startswith('RA') or act.startswith('LA') and not act.endswith('(root)'):
            apply_actions.append(act + f'({label})')
        elif act.startswith('AD'):
            apply_actions.append(act + f'({labelA})')
        else:
            apply_actions.append(act)
    toks = [tok for tok in tokens if tok != "<eof>"]
    tr = Transitions(toks, verbose=False)
    tr.applyActions(apply_actions)
    return tr.amr


class AMRModel(torch.nn.Module):

    def __init__(self, amrs, oracle_transitions, possible_predicates, embedding_dim, action_embedding_dim,
                 char_embedding_dim, hidden_dim, char_hidden_dim, rnn_layers, dropout_ratio,
                 pretrained_dim=1024, experiment=None,
                 use_gpu=False, use_chars=False, use_bert=False, use_attention=False,
                 use_function_words=False, use_function_words_rels=False, parse_unaligned=False,
                 weight_inputs=False, attend_inputs=False):
        super(AMRModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_hidde_dim = char_hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.hidden_dim = hidden_dim
        self.exp = experiment
        self.pretrained_dim = pretrained_dim
        self.rnn_layers = rnn_layers
        self.use_bert = use_bert
        self.use_chars = use_chars
        self.use_attention = use_attention
        self.use_function_words_all = use_function_words
        self.use_function_words_rels = use_function_words_rels
        self.use_function_words = use_function_words or use_function_words_rels
        self.parse_unaligned = parse_unaligned
        self.weight_inputs = weight_inputs
        self.attend_inputs = attend_inputs

        self.warm_up = False

        self.possible_predicates = possible_predicates

        try:
            self.lemmatizer = spacy.load('en', disable=['parser', 'ner'])
            self.lemmatizer.tokenizer = NoTokenizer(self.lemmatizer.vocab)
        except OSError:
            self.lemmatizer = None
            print_log('parser', "Warning: Could not load Spacy English model. Please install with 'python -m spacy download en'.", file=sys.stderr)

        self.state_dim = 3 * hidden_dim + (hidden_dim if use_attention else 0) \
            + (hidden_dim if self.use_function_words_all else 0)

        self.state_size = self.state_dim // hidden_dim

        if self.weight_inputs or self.attend_inputs:
            self.state_dim = hidden_dim

        self.use_gpu = use_gpu

        # Vocab and indices

        self.char2idx = {'<unk>': 0}
        self.word2idx = {'<unk>': 0, '<eof>': 1, '<ROOT>': 2, '<unaligned>': 3}
        self.node2idx = {}
        word_counter = Counter()

        for amr in amrs:
            for tok in amr.tokens:
                word_counter[tok] += 1
                self.word2idx.setdefault(tok, len(self.word2idx))
                for ch in tok:
                    self.char2idx.setdefault(ch, len(self.char2idx))
            for n in amr.nodes:
                self.node2idx.setdefault(amr.nodes[n], len(self.node2idx))

        self.amrs = amrs

        self.singletons = {self.word2idx[w] for w in word_counter if word_counter[w] == 1}
        self.singletons.discard('<unk>')
        self.singletons.discard('<eof>')
        self.singletons.discard('<ROOT>')
        self.singletons.discard('<unaligned>')

        self.labelsO2idx = {'<pad>': 0}
        self.labelsA2idx = {'<pad>': 0}
        self.pred2idx = {'<pad>': 0}
        self.action2idx = {'<pad>': 0}
        for tr in oracle_transitions:
            for a in tr.actions:
                a = Transitions.readAction(a)[0]
                self.action2idx.setdefault(a, len(self.action2idx))
            for p in tr.predicates:
                self.pred2idx.setdefault(p, len(self.pred2idx))
            for l in tr.labels:
                self.labelsO2idx.setdefault(l, len(self.labelsO2idx))
            for l in tr.labelsA:
                self.labelsA2idx.setdefault(l, len(self.labelsA2idx))

        self.vocab_size = len(self.word2idx)
        self.action_size = len(self.action2idx)

        self.labelA_size = len(self.labelsA2idx)
        self.labelO_size = len(self.labelsO2idx)
        self.pred_size = len(self.pred2idx)

        self.idx2labelO = {v: k for k, v in self.labelsO2idx.items()}
        self.idx2labelA = {v: k for k, v in self.labelsA2idx.items()}
        self.idx2node = {v: k for k, v in self.node2idx.items()}
        self.idx2pred = {v: k for k, v in self.pred2idx.items()}
        self.idx2action = {v: k for k, v in self.action2idx.items()}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        # self.ner_map = ner_map
        self.labelsA = []
        for k, v in self.labelsA2idx.items():
            self.labelsA.append(v)
        self.labelsO = []
        for k, v in self.labelsO2idx.items():
            self.labelsO.append(v)
        self.preds = []
        for k, v in self.pred2idx.items():
            self.preds.append(v)
        print_log('parser', f'Number of characters: {len(self.char2idx)}')
        print_log('parser', f'Number of words: {len(self.word2idx)}')
        print_log('parser', f'Number of nodes: {len(self.node2idx)}')
        print_log('parser', f'Number of actions: {len(self.action2idx)}')
        for action in self.action2idx:
            print('\t', action)
        print_log('parser', f'Number of labels: {len(self.labelsO2idx)}')
        print_log('parser', f'Number of labelsA: {len(self.labelsA2idx)}')
        print_log('parser', f'Number of predicates: {len(self.pred2idx)}')

        # Parameters
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.action_embeds = nn.Embedding(self.action_size, action_embedding_dim)
        self.labelA_embeds = nn.Embedding(self.labelA_size, action_embedding_dim)
        self.labelO_embeds = nn.Embedding(self.labelO_size, action_embedding_dim)
        self.pred_embeds = nn.Embedding(self.pred_size, action_embedding_dim)
        self.pred_unk_embed = nn.Parameter(torch.randn(1, self.action_embedding_dim), requires_grad=True)
        self.empty_emb = nn.Parameter(torch.randn(1, hidden_dim), requires_grad=True)

        # Stack-LSTMs
        self.buffer_lstm = nn.LSTMCell(self.embedding_dim, hidden_dim)
        self.stack_lstm = nn.LSTMCell(self.embedding_dim, hidden_dim)
        self.action_lstm = nn.LSTMCell(action_embedding_dim, hidden_dim)
        self.lstm_initial_1 = utils.xavier_init(self.use_gpu, 1, self.hidden_dim)
        self.lstm_initial_2 = utils.xavier_init(self.use_gpu, 1, self.hidden_dim)
        self.lstm_initial = (self.lstm_initial_1, self.lstm_initial_2)

        if self.use_chars:
            self.char_embeds = nn.Embedding(len(self.char2idx), char_embedding_dim)
            self.unaligned_char_embed = nn.Parameter(torch.randn(1, 2 * char_hidden_dim), requires_grad=True)
            self.root_char_embed = nn.Parameter(torch.randn(1, 2 * char_hidden_dim), requires_grad=True)
            self.pad_char_embed = nn.Parameter(torch.zeros(1, 2 * char_hidden_dim))
            self.char_lstm_forward = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=rnn_layers,
                                             dropout=dropout_ratio)
            self.char_lstm_backward = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=rnn_layers,
                                              dropout=dropout_ratio)

            self.tok_2_embed = nn.Linear(self.embedding_dim + 2 * char_hidden_dim, self.embedding_dim)

        if self.use_bert:
            # bert embeddings to LSTM input
            self.pretrained_2_embed = nn.Linear(self.embedding_dim + self.pretrained_dim, self.embedding_dim)

        if use_attention:
            self.forward_lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)
            self.backward_lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)

            self.attention_weights = nn.Parameter(torch.randn(2 * hidden_dim, 2 * hidden_dim), requires_grad=True)

            self.attention_ff1_1 = nn.Linear(2 * hidden_dim, hidden_dim)

        self.dropout_emb = nn.Dropout(p=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.action_softmax1 = nn.Linear(self.state_dim, hidden_dim)
        self.labelA_softmax1 = nn.Linear(self.state_dim, hidden_dim)
        self.pred_softmax1 = nn.Linear(self.state_dim, hidden_dim)
        if not self.use_function_words_rels:
            self.label_softmax1 = nn.Linear(self.state_dim, hidden_dim)
        else:
            self.label_softmax1 = nn.Linear(self.state_dim + hidden_dim, hidden_dim)

        self.action_softmax2 = nn.Linear(hidden_dim, len(self.action2idx) + 2)
        self.labelA_softmax2 = nn.Linear(hidden_dim, len(self.labelsA2idx) + 2)
        self.label_softmax2 = nn.Linear(hidden_dim, len(self.labelsO2idx) + 2)
        self.pred_softmax2 = nn.Linear(hidden_dim, len(self.pred2idx) + 2)

        # composition functions
        self.arc_composition_head = nn.Linear(2 * self.embedding_dim + self.action_embedding_dim, self.embedding_dim)
        self.merge_composition = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.dep_composition = nn.Linear(self.embedding_dim + self.action_embedding_dim, self.embedding_dim)
        self.addnode_composition = nn.Linear(self.embedding_dim + self.action_embedding_dim, self.embedding_dim)
        self.pred_composition = nn.Linear(self.embedding_dim + self.action_embedding_dim, self.embedding_dim)

        # experiments
        if self.use_function_words:
            self.functionword_lstm = nn.LSTMCell(self.embedding_dim, hidden_dim)

        if self.parse_unaligned:
            self.pred_softmax1_unaligned = nn.Linear(self.state_dim, hidden_dim)
            self.pred_softmax2_unaligned = nn.Linear(hidden_dim, len(self.pred2idx) + 2)

        if self.weight_inputs:
            self.action_attention = nn.Parameter(torch.zeros(self.state_size), requires_grad=True)
            self.label_attention = nn.Parameter(torch.zeros(self.state_size), requires_grad=True)
            self.labelA_attention = nn.Parameter(torch.zeros(self.state_size), requires_grad=True)
            self.pred_attention = nn.Parameter(torch.zeros(self.state_size), requires_grad=True)
            if self.parse_unaligned:
                self.pred_attention_unaligned = nn.Parameter(torch.zeros(self.state_size), requires_grad=True)
        elif self.attend_inputs:
            self.action_attention = torch.nn.Linear(self.state_size*2, self.state_size)
            self.label_attention = torch.nn.Linear(self.state_size*2, self.state_size)
            self.labelA_attention = torch.nn.Linear(self.state_size*2, self.state_size)
            self.pred_attention = torch.nn.Linear(self.state_size*2, self.state_size)
            if self.parse_unaligned:
                self.pred_attention_unaligned = torch.nn.Linear(self.state_size*2, self.state_size)
            self.prevent_overfitting = torch.nn.Linear(hidden_dim, self.state_size*2)

        # stats and accuracy
        self.action_acc = utils.Accuracy()
        self.label_acc = utils.Accuracy()
        self.labelA_acc = utils.Accuracy()
        self.pred_acc = utils.Accuracy()

        self.action_confusion_matrix = utils.ConfusionMatrix(self.action2idx.keys())
        self.label_confusion_matrix = utils.ConfusionMatrix(self.labelsO2idx.keys())

        self.action_loss = 0
        self.label_loss = 0
        self.labelA_loss = 0
        self.pred_loss = 0

        self.epoch_loss = 0

        self.rand_init()
        if self.use_gpu:
            for m in self.modules():
                m.cuda()

    def reset_stats(self):
        self.action_acc.reset()
        self.label_acc.reset()
        self.labelA_acc.reset()
        self.pred_acc.reset()

        self.action_confusion_matrix.reset()
        self.label_confusion_matrix.reset()

        self.action_loss = 0
        self.label_loss = 0
        self.labelA_loss = 0
        self.pred_loss = 0
        self.epoch_loss = 0

    def rand_init(self):

        utils.initialize_embedding(self.word_embeds.weight)
        utils.initialize_embedding(self.action_embeds.weight)
        utils.initialize_embedding(self.pred_embeds.weight)
        utils.initialize_embedding(self.labelA_embeds.weight)
        utils.initialize_embedding(self.labelO_embeds.weight)

        utils.initialize_lstm_cell(self.buffer_lstm)
        utils.initialize_lstm_cell(self.action_lstm)
        utils.initialize_lstm_cell(self.stack_lstm)

        if self.use_chars:
            utils.initialize_embedding(self.char_embeds.weight)
            utils.initialize_lstm(self.char_lstm_forward)
            utils.initialize_lstm(self.char_lstm_backward)
            utils.initialize_linear(self.tok_2_embed)
        if self.use_bert:
            utils.initialize_linear(self.pretrained_2_embed)
        if self.use_attention:
            utils.initialize_lstm(self.forward_lstm)
            utils.initialize_lstm(self.backward_lstm)
            utils.initialize_linear(self.attention_ff1_1)

        utils.initialize_linear(self.action_softmax1)
        utils.initialize_linear(self.labelA_softmax1)
        utils.initialize_linear(self.pred_softmax1)
        utils.initialize_linear(self.label_softmax1)
        utils.initialize_linear(self.action_softmax2)
        utils.initialize_linear(self.labelA_softmax2)
        utils.initialize_linear(self.label_softmax2)
        utils.initialize_linear(self.pred_softmax2)

        utils.initialize_linear(self.arc_composition_head)
        utils.initialize_linear(self.merge_composition)
        utils.initialize_linear(self.dep_composition)
        utils.initialize_linear(self.pred_composition)
        utils.initialize_linear(self.addnode_composition)

        if self.use_function_words:
            utils.initialize_lstm_cell(self.functionword_lstm)
        if self.parse_unaligned:
            utils.initialize_linear(self.pred_softmax1_unaligned)
            utils.initialize_linear(self.pred_softmax2_unaligned)
        if self.attend_inputs:
            utils.initialize_zero(self.action_attention)
            utils.initialize_zero(self.label_attention)
            utils.initialize_zero(self.labelA_attention)
            utils.initialize_zero(self.pred_attention)
            if self.parse_unaligned:
                utils.initialize_zero(self.pred_attention_unaligned)
            utils.initialize_linear(self.prevent_overfitting)

    def forward(self, sent_idx, sentence_tensor, labelsO, labelsA, actions, preds, obj,
                tokens=None, bert_embedding=None):

        batch_size = sentence_tensor.size()[0]

        losses = []
        for i in range(batch_size):
            if obj == 'ML':
                loss, _, _, _, _ = self.forward_single(
                    sentence_tensor[i].unsqueeze(0),
                    labelsO[i].unsqueeze(0),
                    labelsA[i].unsqueeze(0), actions[i].unsqueeze(0),
                    preds[i].unsqueeze(0), 'train',
                    tokens[i] if tokens else None,
                    bert_embedding[i] if bert_embedding else None
                )
            else:

                # amr predicted greedily
                _, gr_actions, gr_labels, gr_labelsA, gr_predicates = self.forward_single(
                    sentence_tensor[i].unsqueeze(0),
                    labelsO[i].unsqueeze(0),
                    labelsA[i].unsqueeze(0), actions[i].unsqueeze(0),
                    preds[i].unsqueeze(0), 'predict',
                    tokens[i] if tokens else None,
                    bert_embedding[i] if bert_embedding else None
                )
                greedy_amr = build_amr(tokens[i], gr_actions, gr_labels, gr_labelsA, gr_predicates)

                # amr predicted by sampling
                loss, sm_actions, sm_labels, sm_labelsA, sm_predicates = self.forward_single(
                    sentence_tensor[i].unsqueeze(0),
                    labelsO[i].unsqueeze(0),
                    labelsA[i].unsqueeze(0), actions[i].unsqueeze(0),
                    preds[i].unsqueeze(0), 'sample',
                    tokens[i] if tokens else None,
                    bert_embedding[i] if bert_embedding else None
                )
                sample_amr = build_amr(
                    tokens[i], sm_actions, sm_labels, sm_labelsA, sm_predicates
                )

                # amr predicted by oracle
                gold_amr = self.amrs[sent_idx[i].item()]

                # write to files for scoring
                pid = f'{os.getpid()}'
                prank = str(torch.distributed.get_rank())
                grdy_name = prank + "_" + pid + "_greedy"
                gold_name = prank+"_" + pid + "_gold"
                smpl_name = prank + "_" + pid + "_smpl"
                fgrdy = open(grdy_name + ".amr", 'w')
                fgold = open(gold_name + ".amr", 'w')
                fsmpl = open(smpl_name + ".amr", 'w')
                fgrdy.write(greedy_amr.toJAMRString())
                fsmpl.write(sample_amr.toJAMRString())
                fgold.write(gold_amr.toJAMRString())
                fgrdy.close()
                fgold.close()
                fsmpl.close()
                os.system(f'python smatch/smatch.py --significant 3 -f {gold_name}.amr {grdy_name}.amr > {grdy_name}_smatch.txt')
                os.system(f'python smatch/smatch.py --significant 3 -f {gold_name}.amr {smpl_name}.amr > {smpl_name}_smatch.txt')
                fgrdy_smatch = open(grdy_name+'_smatch.txt')
                inputs = fgrdy_smatch.readline().strip().split()
                greedy_smatch = float(inputs[-1]) if len(inputs) > 1 else 0
                fsmpl_smatch = open(smpl_name+'_smatch.txt')
                inputs = fsmpl_smatch.readline().strip().split()
                sample_smatch = float(inputs[-1]) if len(inputs) > 1 else 0

                loss *= (sample_smatch - greedy_smatch)

                fgrdy_smatch.close()
                fsmpl_smatch.close()
                os.remove(grdy_name+".amr")
                os.remove(gold_name+".amr")
                os.remove(smpl_name+".amr")
                os.remove(grdy_name+"_smatch.txt")
                os.remove(smpl_name+"_smatch.txt")

            losses.append(loss.reshape((1,)))
        losses = torch.cat(losses, 0)
        return losses.sum()

    def forward_single(self, sentence_tensor, labelsO=None, labelsA=None, actions=None, preds=None, mode='train',
                       tokens=None, bert_embedding=None):

        word_embeds = self.dropout_emb(self.word_embeds(sentence_tensor))
        word_embeds = word_embeds.squeeze(0)
        if mode == 'train':
            action_embeds = self.dropout_emb(self.action_embeds(actions))
            action_embeds = action_embeds.squeeze(0)
            actions = actions.squeeze(0)
            pred_embeds = self.dropout_emb(self.pred_embeds(preds))
            pred_embeds = pred_embeds.squeeze(0)
            preds = preds.squeeze(0)
            labelA_embeds = self.dropout_emb(self.labelA_embeds(labelsA))
            labelA_embeds = labelA_embeds.squeeze(0)
            labelsA = labelsA.squeeze(0)
            labelO_embeds = self.dropout_emb(self.labelO_embeds(labelsO))
            labelO_embeds = labelO_embeds.squeeze(0)
            labelsO = labelsO.squeeze(0)
        else:
            # a hack that assigns embeddings instead of tensors (at prediciton time)
            action_embeds = self.action_embeds
            pred_embeds = self.pred_embeds
            labelA_embeds = self.labelA_embeds
            labelO_embeds = self.labelO_embeds

        sentence_tensor = sentence_tensor.squeeze(0)
        action_count = 0

        buffer = sl.StackRNN(self.buffer_lstm, self.lstm_initial, self.dropout, torch.tanh, self.empty_emb)
        stack = sl.StackRNN(self.stack_lstm, self.lstm_initial, self.dropout, torch.tanh, self.empty_emb)
        action = sl.StackRNN(self.action_lstm, self.lstm_initial, self.dropout, torch.tanh, self.empty_emb)
        latent = []

        if self.use_function_words:
            functionwords = sl.StackRNN(self.functionword_lstm, self.lstm_initial, self.dropout, torch.tanh,
                                        self.empty_emb)

        total_losses = []
        action_losses = []
        label_losses = []
        labelA_losses = []
        pred_losses = []
        sentence_array = sentence_tensor.tolist()
        token_embedding = list()

        # build word embeddings (using characters, bert, linear layers, etc.)
        for word_idx, word in enumerate(sentence_array):
            if self.use_chars:
                if self.idx2word[word] in ['<eof>', '']:
                    tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), self.pad_char_embed], 1)
                elif self.idx2word[word] == '<ROOT>':
                    tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), self.root_char_embed], 1)
                elif self.idx2word[word] == '<unaligned>':
                    tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), self.unaligned_char_embed], 1)
                else:
                    chars_in_word = [self.char2idx[char] if char in self.char2idx else self.char2idx['<unk>'] for char
                                     in tokens[word_idx]]
                    chars_tensor = torch.LongTensor(chars_in_word)
                    if self.use_gpu:
                        chars_tensor = chars_tensor.cuda()
                    chars_embeds = self.char_embeds(chars_tensor.unsqueeze(0))
                    char_fw, _ = self.char_lstm_forward(chars_embeds.transpose(0, 1))
                    char_bw, _ = self.char_lstm_backward(utils.reverse_sequence(chars_embeds.transpose(0, 1), gpu=self.use_gpu))
                    bw = char_bw[-1, :, :]
                    fw = char_fw[-1, :, :]

                    tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), fw, bw], 1)
                tok_rep = torch.tanh(self.tok_2_embed(tok_rep))
            else:
                tok_rep = word_embeds[word_idx].unsqueeze(0)

            if self.use_bert and self.idx2word[word] not in ['<ROOT>', '<unaligned>', '<eof>']:
                bert_embed = torch.from_numpy(bert_embedding[word_idx]).float()
                if self.use_gpu:
                    bert_embed = bert_embed.cuda()
                tok_rep = torch.cat([tok_rep, bert_embed.unsqueeze(0)], 1)
                tok_rep = self.pretrained_2_embed(tok_rep)

            if word_idx == 0:
                token_embedding = tok_rep
            elif sentence_array[word_idx] != 1:
                token_embedding = torch.cat([token_embedding, tok_rep], 0)

        bufferint = []
        stackint = []
        latentint = []

        # add tokens to buffer
        i = len(sentence_array) - 1
        for idx, token in zip(reversed(sentence_array), reversed(tokens)):
            if self.idx2word[idx] != '<eof>':
                tok_embed = token_embedding[i].unsqueeze(0)
                if self.idx2word[idx] == '<unaligned>' and self.parse_unaligned:
                    latent.append((tok_embed, token))
                    latentint.append(idx)
                else:
                    buffer.push(tok_embed, token)
                    bufferint.append(idx if token != '<ROOT>' else -1)
            i -= 1

        predict_actions = []
        predict_labels = []
        predict_labelsA = []
        predict_predicates = []

        # vars for testing valid actions
        is_entity = set()
        has_root = False
        is_edge = set()
        is_confirmed = set()
        is_functionword = set(bufferint)
        is_functionword.discard(-1)
        swapped_words = {}
        lemmas = None

        if self.use_attention:
            forward_output, _ = self.forward_lstm(token_embedding.unsqueeze(1))
            backward_output, _ = self.backward_lstm(utils.reverse_sequence(token_embedding, gpu=self.use_gpu).unsqueeze(1))
            backward_output = utils.reverse_sequence(backward_output, gpu=self.use_gpu)

        while not (len(bufferint) == 0 and len(stackint) == 0):

            label = ''
            labelA = ''
            predicate = ''

            extra = []
            if self.use_attention:
                last_state = torch.cat([action.output(), stack.output()], 1)

                x = torch.cat([forward_output, backward_output], 2).squeeze(1)
                y = torch.matmul(self.attention_weights, last_state.squeeze(0))
                attention = torch.softmax(torch.matmul(x, y), dim=0)
                attention_output = torch.matmul(x.transpose(0, 1), attention).unsqueeze(0)

                attention_output = torch.tanh(self.attention_ff1_1(attention_output))

                extra += [attention_output]
            if self.use_function_words_all:
                extra += [functionwords.output()]

            lstms_output = [stack.output(), buffer.output(), action.output()] + extra

            if not (self.weight_inputs or self.attend_inputs):
                lstms_output = torch.cat(lstms_output, 1)

            if self.attend_inputs:
                stack_state = torch.tanh(self.prevent_overfitting(self.dropout(stack.output())))

            future_label_loss = []

            if mode == 'train':
                valid_actions = list(self.action2idx.values())
            else:
                extra = functionwords.output() if self.use_function_words_rels else None
                if self.weight_inputs:
                    inputs = self.weight_vectors(self.label_attention, lstms_output)
                elif self.attend_inputs:
                    inputs = self.weight_vectors(self.label_attention(stack_state).squeeze(), lstms_output)
                else:
                    inputs = lstms_output

                label_embedding, label, future_label_loss, correct = self.predict_with_softmax(
                    self.label_softmax1, self.label_softmax2, inputs, labelsO, self.labelsO,
                    labelO_embeds, self.idx2labelO, 'predict', future_label_loss, action_count, extra=extra)
                valid_actions = self.get_possible_actions(stackint, bufferint, latentint, has_root, is_entity, is_edge,
                                                          label, is_confirmed, swapped_words)

            if self.weight_inputs:
                inputs = self.weight_vectors(self.action_attention, lstms_output)
            elif self.attend_inputs:
                inputs = self.weight_vectors(self.action_attention(stack_state).squeeze(), lstms_output)
            else:
                inputs = lstms_output
            act_embedding, real_action, action_losses, correct = self.predict_with_softmax(self.action_softmax1,
                                                                                           self.action_softmax2,
                                                                                           inputs, actions,
                                                                                           valid_actions, action_embeds,
                                                                                           self.idx2action, mode,
                                                                                           action_losses, action_count,
                                                                                           self.action_confusion_matrix)
            if mode == 'train':
                self.action_acc.add(correct)

            # Perform Action
            action.push(act_embedding, real_action)

            # SHIFT
            if real_action.startswith('SH'):

                buffer0 = buffer.pop()
                if len(bufferint) == 0 or not buffer0:
                    # CLOSE
                    bufferint = []
                    stackint = []
                    predict_actions.append(real_action)
                    predict_labels.append(label)
                    predict_labelsA.append(labelA)
                    predict_predicates.append(predicate)
                    break
                s0 = bufferint.pop()
                stackint.append(s0)
                tok_buffer_embedding, buffer_token = buffer0
                stack.push(tok_buffer_embedding, buffer_token)
            # REDUCE
            elif real_action.startswith('RE'):
                s0 = stackint.pop()
                tok_stack0_embedding, stack0_token = stack.pop()
                if self.use_function_words and s0 in is_functionword:
                    functionwords.push(tok_stack0_embedding, stack0_token)
            # UNSHIFT
            elif real_action.startswith('UN'):
                s0 = stackint.pop()
                s1 = stackint.pop()
                bufferint.append(s1)
                stackint.append(s0)
                tok_stack0_embedding, stack0_token = stack.pop()
                tok_stack1_embedding, stack1_token = stack.pop()
                stack.push(tok_stack0_embedding, stack0_token)
                buffer.push(tok_stack1_embedding, stack1_token)
                if s1 not in swapped_words:
                    swapped_words[s1] = []
                swapped_words[s1].append(s0)
            # MERGE
            elif real_action.startswith('ME'):
                s0 = stackint.pop()
                tok_stack0_embedding, stack0_token = stack.pop()
                tok_stack1_embedding, stack1_token = stack.pop()
                head_embedding = torch.tanh(
                    self.merge_composition(torch.cat((tok_stack0_embedding, tok_stack1_embedding), dim=1)))
                stack.push(head_embedding, stack1_token)
                is_functionword.discard(stackint[-1])
            # DEPENDENT
            elif real_action.startswith('DE'):
                tok_stack0_embedding, stack0_token = stack.pop()
                head_embedding = torch.tanh(
                    self.dep_composition(torch.cat((tok_stack0_embedding, act_embedding), dim=1)))
                stack.push(head_embedding, stack0_token)
                is_functionword.discard(stackint[-1])
            # RA
            elif real_action.startswith('RA'):
                if mode == 'train' or mode == 'sample':
                    extra = functionwords.output() if self.use_function_words_rels else None
                    if self.weight_inputs:
                        inputs = self.weight_vectors(self.label_attention, lstms_output)
                    elif self.attend_inputs:
                        inputs = self.weight_vectors(self.label_attention(stack_state).squeeze(), lstms_output)
                    else:
                        inputs = lstms_output
                    label_embedding, label, label_losses, correct = self.predict_with_softmax(
                        self.label_softmax1, self.label_softmax2, inputs, labelsO, self.labelsO,
                        labelO_embeds, self.idx2labelO, mode, label_losses, action_count, self.label_confusion_matrix,
                        extra=extra)
                    self.label_acc.add(correct)
                tok_stack0_embedding, stack0_token = stack.pop()
                tok_stack1_embedding, stack1_token = stack.pop()
                head_embedding = torch.tanh(self.arc_composition_head(
                    torch.cat((tok_stack1_embedding, label_embedding, tok_stack0_embedding), dim=1)))
                dep_embedding = tok_stack0_embedding
                stack.push(head_embedding, stack1_token)
                stack.push(dep_embedding, stack0_token)
                if label == 'root':
                    has_root = True
                is_edge.add((stackint[-2], label, stackint[-1]))
                if label.startswith('ARG') and (not label.endswith('of')):
                    is_edge.add((stackint[-2], label))
                is_functionword.discard(stackint[-1])
                is_functionword.discard(stackint[-2])
            # LA
            elif real_action.startswith('LA'):
                if mode == 'train' or mode == 'sample':
                    extra = functionwords.output() if self.use_function_words_rels else None
                    if self.weight_inputs:
                        inputs = self.weight_vectors(self.label_attention, lstms_output)
                    elif self.attend_inputs:
                        inputs = self.weight_vectors(self.label_attention(stack_state).squeeze(), lstms_output)
                    else:
                        inputs = lstms_output
                    label_embedding, label, label_losses, correct = self.predict_with_softmax(
                        self.label_softmax1, self.label_softmax2, inputs, labelsO, self.labelsO,
                        labelO_embeds, self.idx2labelO, mode, label_losses, action_count, self.label_confusion_matrix,
                        extra=extra)
                    self.label_acc.add(correct)
                tok_stack0_embedding, stack0_token = stack.pop()
                tok_stack1_embedding, stack1_token = stack.pop()
                head_embedding = torch.tanh(self.arc_composition_head(
                    torch.cat((tok_stack0_embedding, label_embedding, tok_stack1_embedding), dim=1)))
                dep_embedding = tok_stack1_embedding
                stack.push(dep_embedding, stack1_token)
                stack.push(head_embedding, stack0_token)
                if label == 'root':
                    has_root = True
                is_edge.add((stackint[-1], label, stackint[-2]))
                if label.startswith('ARG') and (not label.endswith('of')):
                    is_edge.add((stackint[-1], label))
                is_functionword.discard(stackint[-1])
                is_functionword.discard(stackint[-2])
            # PRED
            elif real_action.startswith('PRED'):
                tok = stack.last()[1]
                pred_softmax1 = self.pred_softmax1
                pred_softmax2 = self.pred_softmax2
                if self.weight_inputs or self.attend_inputs:
                    pred_attention = self.pred_attention
                if self.parse_unaligned and tok == '<unaligned>':
                    pred_softmax1 = self.pred_softmax1_unaligned
                    pred_softmax2 = self.pred_softmax2_unaligned
                    if self.weight_inputs or self.attend_inputs:
                        pred_attention = self.pred_attention_unaligned

                if self.weight_inputs:
                    inputs = self.weight_vectors(pred_attention, lstms_output)
                elif self.attend_inputs:
                    inputs = self.weight_vectors(pred_attention(stack_state).squeeze(), lstms_output)
                else:
                    inputs = lstms_output

                possible_predicates = []
                if tok in self.possible_predicates:
                    for p in self.possible_predicates[tok]:
                        if p in self.pred2idx:
                            possible_predicates.append(self.pred2idx[p])

                if mode == 'train' and preds[action_count].item() not in possible_predicates:
                    possible_predicates = self.preds

                if not possible_predicates or stackint[-1] == self.word2idx['<unk>']:
                    pred_embedding = self.pred_unk_embed
                    # FIXME: Temporary fix for tok not in tokens killing threads
                    if self.lemmatizer is not None and tok in tokens:
                        if not lemmas:
                            lemmas = self.lemmatizer(tokens)
                        lemma = lemmas[tokens.index(tok)].lemma_
                        predicate = lemma
                    else:
                        predicate = tok
                else:
                    pred_embedding, predicate, pred_losses, correct = self.predict_with_softmax(
                        pred_softmax1, pred_softmax2, inputs, preds, possible_predicates,
                        pred_embeds, self.idx2pred, mode, pred_losses, action_count)

                if mode == 'train':
                    self.pred_acc.add(correct)

                tok_stack0_embedding, stack0_token = stack.pop()
                pred_embedding = torch.tanh(
                    self.pred_composition(torch.cat((pred_embedding, tok_stack0_embedding), dim=1)))
                stack.push(pred_embedding, predicate)
                is_confirmed.add(stackint[-1])
                is_functionword.discard(stackint[-1])
            # ADDNODE
            elif real_action.startswith('AD'):
                if self.weight_inputs:
                    inputs = self.weight_vectors(self.labelA_attention, lstms_output)
                elif self.attend_inputs:
                    inputs = self.weight_vectors(self.labelA_attention(stack_state).squeeze(), lstms_output)
                else:
                    inputs = lstms_output
                labelA_embedding, labelA, labelA_losses, correct = self.predict_with_softmax(
                    self.labelA_softmax1, self.labelA_softmax2, inputs, labelsA, self.labelsA,
                    labelA_embeds, self.idx2labelA, mode, labelA_losses, action_count)
                if mode == 'train':
                    self.labelA_acc.add(correct)
                tok_stack0_embedding, stack0_token = stack.pop()
                head_embedding = torch.tanh(
                    self.addnode_composition(torch.cat((tok_stack0_embedding, labelA_embedding), dim=1)))
                stack.push(head_embedding, stack0_token)
                is_entity.add(stackint[-1])
                is_functionword.discard(stackint[-1])
            # INTRODUCE
            elif real_action.startswith('IN'):
                tok_latent_embedding, latent_token = latent.pop()
                stack.push(tok_latent_embedding, latent_token)
                item = latentint.pop()
                stackint.append(item)

            predict_actions.append(real_action)
            predict_labels.append(label)
            predict_labelsA.append(labelA)
            predict_predicates.append(predicate)
            action_count += 1
            if mode == 'train' and (
                    actions[action_count].item() == 0 or (actions[action_count].item() == 1 and len(bufferint) == 0)):
                bufferint = []
                stackint = []
            if action_count > 500:
                bufferint = []
                stackint = []

        for l in [action_losses, label_losses, labelA_losses, pred_losses]:
            total_losses.extend(l)
        if len(total_losses) > 0:
            total_losses = -torch.sum(torch.stack(total_losses))
        else:
            total_losses = -1

        losses_per_component = []
        for i, loss in enumerate([action_losses, label_losses, labelA_losses, pred_losses]):
            if len(loss) > 0:
                losses_per_component.append(-torch.sum(torch.stack(loss)).item())
            else:
                losses_per_component.append(0)

        self.action_loss += losses_per_component[0]
        self.label_loss += losses_per_component[1]
        self.labelA_loss += losses_per_component[2]
        self.pred_loss += losses_per_component[3]
        return total_losses, predict_actions, predict_labels, predict_labelsA, predict_predicates

    # return embedding, string and update losses (if it is in training mode)
    def predict_with_softmax(self, softmax1, softmax2, lstms_output, tensor, elements, embeds, idx2map, mode,
                             losses, action_count, confusion_matrix=None, extra=None):
        if extra is not None:
            hidden_output = torch.tanh(softmax1(self.dropout(torch.cat((lstms_output, extra), dim=1))))
        else:
            hidden_output = torch.tanh(softmax1(self.dropout(lstms_output)))

        if self.use_gpu is True:
            logits = softmax2(hidden_output)[0][torch.LongTensor(elements).cuda()]
        else:
            logits = softmax2(hidden_output)[0][torch.LongTensor(elements)]
        tbl = {a: i for i, a in enumerate(elements)}
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)

        # if mode is sample then dont pick the argmax
        idx = 0
        if mode != 'sample':
            idx = log_probs.argmax().item()
        else:
            log_dist = log_probs
            if random.randint(1, 20) == 1:
                log_dist = torch.mul(log_dist, 0.5)
            dist = torch.distributions.categorical.Categorical(logits=log_dist)
            idx = dist.sample()
            losses.append(log_probs[idx])

        predict = elements[idx]
        pred_string = idx2map[predict]
        if mode == 'train':
            if log_probs is not None:
                losses.append(log_probs[tbl[tensor[action_count].item()]])
            gold_string = idx2map[tensor[action_count].item()]
            embedding = embeds[action_count].unsqueeze(0)
            if confusion_matrix:
                confusion_matrix.add(gold_string, pred_string)
            return embedding, gold_string, losses, (pred_string == gold_string)
        elif mode == 'predict' or mode == 'sample':
            predict_tensor = torch.from_numpy(np.array([predict])).cuda() if self.use_gpu else torch.from_numpy(
                np.array([predict]))
            embeddings = self.dropout_emb(embeds(predict_tensor))
            embedding = embeddings[0].unsqueeze(0)
            return embedding, pred_string, losses, False

    def weight_vectors(self, weights, vecs):
        if not self.warm_up:
            weights = self.state_size * F.softmax(weights, dim=0)
            for i, vec in enumerate(vecs):
                vecs[i] = torch.mul(weights[i], vec)
        return torch.cat(vecs, 0).sum(dim=0).unsqueeze(0)

    def get_possible_actions(self, stackint, bufferint, latentint, has_root, is_entity, is_edge, label, is_confirmed,
                             swapped_words):
        valid_actions = []
        for k, v in self.action2idx.items():
            if k.startswith('SH') or k.startswith('CL'):
                if len(bufferint) == 0 and len(stackint) > 5:
                    continue
                valid_actions.append(v)
            elif k.startswith('RE'):
                if len(stackint) >= 1:
                    valid_actions.append(v)
            elif k.startswith('PR'):
                if len(stackint) >= 1 and stackint[-1] not in is_confirmed and stackint[-1] != -1:
                    valid_actions.append(v)
            elif k.startswith('UN'):
                if len(stackint) >= 2:
                    stack0 = stackint[-1]
                    stack1 = stackint[-2]
                    if stack0 in swapped_words and stack1 in swapped_words[stack0]:
                        continue
                    if stack1 in swapped_words and stack0 in swapped_words[stack1]:
                        continue
                    valid_actions.append(v)
            elif k == '<pad>':
                continue
            elif k.startswith('LA') or k.startswith('RA'):
                if k.endswith('(root)') and has_root:
                    continue
                if len(stackint) >= 2:
                    if (stackint[-1] == -1 or stackint[-2] == -1) and not k.endswith('(root)'):
                        continue
                    if label == 'root' and not k.endswith('(root)'):
                        continue
                    if k.startswith('LA') and (stackint[-1], label, stackint[-2]) in is_edge:
                        continue
                    if k.startswith('RA') and (stackint[-2], label, stackint[-1]) in is_edge:
                        continue
                    if k.startswith('LA') and (stackint[-1], label) in is_edge:
                        continue
                    if k.startswith('RA') and (stackint[-2], label) in is_edge:
                        continue
                    valid_actions.append(v)
            elif k.startswith('IN'):
                if len(latentint) >= 1:
                    valid_actions.append(v)
            elif k.startswith('DE'):
                if len(stackint) >= 1 and stackint[-1] != -1:
                    valid_actions.append(v)
            elif k.startswith('AD'):
                if len(stackint) >= 1 and stackint[-1] not in is_entity and stackint[-1] != -1:
                    valid_actions.append(v)
            elif k.startswith('ME'):
                if len(stackint) >= 2 and stackint[-1] != -1 and stackint[-2] != -1:
                    valid_actions.append(v)
            else:
                raise Exception(f'Unrecognized Action: {k}')
        return valid_actions


if __name__ == '__main__':
    main()
