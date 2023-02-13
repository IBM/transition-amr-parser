# Standalone AMR parser

import os
import json
import torch
from transition_amr_parser.model import AMRModel
import transition_amr_parser.utils as utils
from fairseq.models.roberta import RobertaModel
from transition_amr_parser.roberta_utils import extract_features_aligned_to_words


class AMRParser():

    def __init__(self, model_path, roberta_cache_path=None, oracle_stats_path=None, config_path=None, model_use_gpu=False, roberta_use_gpu=False, verbose=False, logger=None):
        if not oracle_stats_path:
            model_folder = os.path.dirname(model_path)
            oracle_stats_path = os.path.join(model_folder, "train.rules.json")
            assert os.path.isfile(oracle_stats_path), \
                f'Expected train.rules.json in {model_folder}'
        if not config_path:
            model_folder = os.path.dirname(model_path)
            config_path = os.path.join(model_folder, "config.json")
            assert os.path.isfile(config_path), \
                f'Expected config.json in {model_folder}'
        self.model = self.load_model(model_path, oracle_stats_path, config_path, model_use_gpu)
        self.roberta = self.load_roberta(roberta_use_gpu, roberta_cache_path)
        self.logger = logger

    def load_roberta(self, roberta_use_gpu, roberta_cache_path=None):

        if not roberta_cache_path:
            # Load the Roberta Model from torch hub
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        else:
            roberta = RobertaModel.from_pretrained(roberta_cache_path, checkpoint_file='model.pt')
        roberta.eval()
        if roberta_use_gpu:
            roberta.cuda()
        return roberta

    def load_model(self, model_path, oracle_stats_path, config_path, model_use_gpu):

        oracle_stats = json.load(open(oracle_stats_path))
        config = json.load(open(config_path))
        model = AMRModel(
            oracle_stats=oracle_stats,
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
