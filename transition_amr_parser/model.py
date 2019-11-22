import time
from datetime import datetime, timedelta
import warnings
from collections import Counter
import argparse
import os
import sys
import random

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
import spacy

from transition_amr_parser.amr import JAMR_CorpusReader
from transition_amr_parser.state_machine import AMRStateMachine
import transition_amr_parser.stack_lstm as sl
import transition_amr_parser.utils as utils
from transition_amr_parser.utils import print_log, smatch_wrapper
from transition_amr_parser.data_oracle import AMR_Oracle

class AMRModel(torch.nn.Module):

    def __init__(self, oracle_stats, embedding_dim, action_embedding_dim,
                 char_embedding_dim, hidden_dim, char_hidden_dim, rnn_layers, dropout_ratio,
                 pretrained_dim=1024, amrs=None, experiment=None,
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

        self.possible_predicates = oracle_stats["possible_predicates"]

        try:
            self.lemmatizer = spacy.load('en', disable=['parser', 'ner'])
            self.lemmatizer.tokenizer = utils.NoTokenizer(self.lemmatizer.vocab)
        except OSError:
            self.lemmatizer = None
            print_log('parser', "Warning: Could not load Spacy English model. Please install with 'python -m spacy download en'.")

        self.state_dim = 3 * hidden_dim + (hidden_dim if use_attention else 0) \
            + (hidden_dim if self.use_function_words_all else 0)

        self.state_size = self.state_dim // hidden_dim

        if self.weight_inputs or self.attend_inputs:
            self.state_dim = hidden_dim

        self.use_gpu = use_gpu

        # Vocab and indices

        self.char2idx = oracle_stats['char2idx']
        self.word2idx = oracle_stats['word2idx']
        self.node2idx = oracle_stats['node2idx']
        word_counter = oracle_stats['word_counter']
        
        self.amrs = amrs

        self.singletons = {self.word2idx[w] for w in word_counter if word_counter[w] == 1}
        self.singletons.discard('<unk>')
        self.singletons.discard('<eof>')
        self.singletons.discard('<ROOT>')
        self.singletons.discard('<unaligned>')

        self.labelsO2idx = oracle_stats["labelsO2idx"]
        self.labelsA2idx = oracle_stats["labelsA2idx"]
        self.pred2idx = oracle_stats["pred2idx"]
        self.action2idx = oracle_stats["action2idx"] 

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
                greedy_amr = self.build_amr(tokens[i], gr_actions, gr_labels, gr_labelsA, gr_predicates)

                # amr predicted by sampling
                loss, sm_actions, sm_labels, sm_labelsA, sm_predicates = self.forward_single(
                    sentence_tensor[i].unsqueeze(0),
                    labelsO[i].unsqueeze(0),
                    labelsA[i].unsqueeze(0), actions[i].unsqueeze(0),
                    preds[i].unsqueeze(0), 'sample',
                    tokens[i] if tokens else None,
                    bert_embedding[i] if bert_embedding else None
                )
                sample_amr = self.build_amr(
                    tokens[i], sm_actions, sm_labels, sm_labelsA, sm_predicates
                )

                # TODO: this will still be neededa unless we provide the AMR as
                # an argument to this function (not for now)
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

    def build_amr(self, tokens, actions, labels, labelsA, predicates):
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
        tr = AMRStateMachine(toks, verbose=False)
        tr.applyActions(apply_actions)
        return tr.amr

    def parse_sentence(self, tokens, sent_rep, bert_emb):
        _, actions, labels, labelsA, predicates = self.forward_single(
            sent_rep,
            mode='predict',
            tokens=tokens,
            bert_embedding=bert_emb
        )
        return self.build_amr(tokens, actions, labels, labelsA, predicates)
