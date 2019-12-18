import socket
import time
import json
from datetime import datetime, timedelta
import warnings
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
from tqdm import tqdm
import h5py

from transition_amr_parser.amr import JAMR_CorpusReader
from transition_amr_parser.state_machine import AMRStateMachine
import transition_amr_parser.utils as utils
from transition_amr_parser.utils import print_log, smatch_wrapper
from transition_amr_parser.data_oracle import AMR_Oracle
from transition_amr_parser.model import AMRModel

warnings.simplefilter(action='ignore', category=FutureWarning)


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser.')
    parser.add_argument("-A", "--amr_training_data", required=True)
    parser.add_argument("-a", "--amr_dev_data", required=True)
    parser.add_argument("-s", "--oracle_stats", required=False)
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

    # TODO: Provide option to precompute outside of this code.
    # 1. Store word/action dictionaries in the rule_stats json that the command line
    # 2. possiblePredicates is already store there
    # This should eliminate the train_amrs and # oracle.transitions
    # dependencies (with the esception of RL mode, which is optional)

    # Oracle computation
    cr = JAMR_CorpusReader()
    cr.load_amrs(args.amr_training_data, training=True)
    cr.load_amrs(args.amr_dev_data, training=False)

    oracle = AMR_Oracle(verbose=False)

    add_unaligned = 10 if args.parse_unaligned else 0
    if args.read_gold_actions:
        oracle.read_actions(args.read_gold_actions)
    else:
        oracle.runOracle(
            cr.amrs,
            add_unaligned=add_unaligned,
            out_actions=args.write_gold_actions
        )

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

    if args.oracle_stats:
        print_log('parser', 'Using pre-computed stats')
        oracle_stats = json.load(open(args.oracle_stats))
    else:
        oracle_stats = oracle.stats

    model = AMRModel(amrs=train_amrs,
                     oracle_stats=oracle_stats,
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
        tr = AMRStateMachine(tokens, verbose=False)
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
        print_log('parser', f'Warning: Distributed processing unavailable. Defaulting to single process.')
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


if __name__ == '__main__':
    main()
