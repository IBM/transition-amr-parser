# Standalone AMR parser

import time
from datetime import timedelta
import os
import signal
import socket
import argparse
from collections import Counter, defaultdict
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from transition_amr_parser.state_machine import AMRStateMachine
from transition_amr_parser.utils import yellow_font
from transition_amr_parser.io import (
    writer,
    read_sentences,
)
from transition_amr_parser.model import AMRModel
import transition_amr_parser.utils as utils
from transition_amr_parser.utils import print_log
import math

from fairseq.models.roberta import RobertaModel
from transition_amr_parser.roberta_utils import extract_features_aligned_to_words

class AMRParser():

    def __init__(self, model_path, oracle_stats_path=None, config_path=None, model_use_gpu=False, roberta_use_gpu=False, verbose=False, logger=None):
        if not oracle_stats_path:
            oracle_stats_path = os.path.join(os.path.dirname(__file__), "train.rules.json")
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.model = self.load_model(model_path, oracle_stats_path, config_path, model_use_gpu)
        self.roberta = self.load_roberta(roberta_use_gpu)
        self.logger = logger 

    def load_roberta(self, roberta_use_gpu):
        
        # Load the Roberta Model
        start = time.time()
        
        RobertaModel = torch.hub.load('pytorch/fairseq', 'roberta.large')
        RobertaModel.eval()
        if roberta_use_gpu:
            RobertaModel.cuda()
        return RobertaModel

    def load_model(self, model_path, oracle_stats_path, config_path, model_use_gpu):

        oracle_stats = json.load(open(oracle_stats_path))
        config = json.load(open(config_path))
        model = AMRModel(
                     oracle_stats = oracle_stats,
                     embedding_dim=config["embedding_dim"],
                     action_embedding_dim=config["action_embedding_dim"],
                     char_embedding_dim=config["char_embedding_dim"],
                     hidden_dim=config["hidden_dim"],
                     char_hidden_dim=config["char_hidden_dim"],
                     rnn_layers=config["rnn_layers"],
                     dropout_ratio=config["dropout_ratio"],
                     pretrained_dim=config["pretrained_dim"],
                     use_bert=config["use_bert"],
                     use_gpu=model_use_gpu,
                     use_chars=config["use_chars"],
                     use_attention=config["use_attention"],
                     use_function_words=config["use_function_words"],
                     use_function_words_rels=config["use_function_words_rels"],
                     parse_unaligned=config["parse_unaligned"],
                     weight_inputs=config["weight_inputs"],
                     attend_inputs=config["attend_inputs"]
                     )

        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def get_embeddings(self, tokens):
        features = extract_features_aligned_to_words(self.roberta, tokens=tokens, use_all_layers=True, return_all_hiddens=True)
        embeddings = []
        for tok in features:
            if str(tok) not in ['<s>', '</s>']:
                embeddings.append(tok.vector)
        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        return embeddings

    def parse_sentence(self, tokens):
        # The model expects <ROOT> token at the end of the input sentence
        if tokens[-1] != "<ROOT>":
            tokens.append("<ROOT>")
        sent_rep = utils.vectorize_words(self.model, tokens, training=False, gpu=self.model.use_gpu)
        bert_emb = self.get_embeddings(tokens)
        amr = self.model.parse_sentence(tokens, sent_rep, bert_emb)
        return amr
