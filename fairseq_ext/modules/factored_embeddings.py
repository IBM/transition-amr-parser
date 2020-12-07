import re
from collections import defaultdict

import torch


class FactoredEmbeddings(torch.nn.Module):

    def __init__(self, vocabulary, embed_dim, action_factoring='base'):

        # register
        super(FactoredEmbeddings, self).__init__()

        self.full_to_factor_map, factor_voc_by_pos = \
            self._get_factor_vocabularies(vocabulary, action_factoring)

        self.decoder_embed_tokens_by_pos = {}
        for position in factor_voc_by_pos.keys():
            num_embeddings = len(factor_voc_by_pos[position])
            self.decoder_embed_tokens_by_pos[position] = torch.nn.Embedding(
                num_embeddings,
                embed_dim,
                num_embeddings - 1 if position > 0 else vocabulary.pad()
            )
            # register map from original to factor indices
            # FIXME: This does not push to CUDA, why?
#             super(FactoredEmbeddings, self).register_buffer(
#                 f'index_map',
#                 self.full_to_factor_map[position]
#             )
            self.full_to_factor_map[position] = self.full_to_factor_map[position].cuda().long()
            # register emebddings for factors
            super(FactoredEmbeddings, self).add_module(
                f'emb_factor{position}',
                self.decoder_embed_tokens_by_pos[position]
            )
        self.factor_voc_by_pos = factor_voc_by_pos
        self.voc = vocabulary
        # self.embeddings = embeddings
        self.embedding_dim = embed_dim
        self.padding_idx = vocabulary.pad()

    def forward(self, indices):

#         # sanity check:
#         for selected_idx in list(set(indices.view(-1).tolist())):
#             rec_action = []
#             for pos in sorted(self.full_to_factor_map.keys()):
#                 if self.full_to_factor_map[pos][selected_idx].item() == 0:
#                     break
#                 i = self.full_to_factor_map[pos][selected_idx]
#                 rec_action.append(self.factor_voc_by_pos[pos][i])
#             if rec_action != self._get_action_factors(self.voc[selected_idx]):
#                 import ipdb; ipdb.set_trace(context=30)
#                 print()

        # breakpoint()
        # Base factor e.g. SHIFT, PRED
        indices2 = self.full_to_factor_map[0][indices]
        output = self.decoder_embed_tokens_by_pos[0](indices2)

        # Add rest of the factors e.g. ARG0
        for pos in sorted(self.full_to_factor_map.keys())[1:]:
            indices2 = self.full_to_factor_map[pos][indices]
            # Ignore padded elements (for factors is allways zero)
            # output[indices2>0] += \
            #    self.decoder_embed_tokens_by_pos[pos](indices2)[indices2>0]
            output += (indices2>0).unsqueeze(2).type(output.type()) * \
                self.decoder_embed_tokens_by_pos[pos](indices2)

        return output

    def _get_factor_vocabularies(self, vocabulary, action_factoring):

        decomposed_actions = []
        group_index_map = defaultdict(dict)
        group_vocabulary = defaultdict(list)
        pad_symbol = vocabulary.symbols[vocabulary.pad()]

        # initialize a mapping array from the original vocabulary to the factor
        # one
        factors = set()
        for index, action in enumerate(vocabulary.symbols):

            # type of action factors
            if action_factoring == 'base':
                action_factors = self._get_action_factors(action)
            elif action_factoring == 'super':
                action_factors = self._get_super_action_factors(action)
            else:
                raise Exception(f'Unknown action factor criteria {action_factors}')

            for position, factor in enumerate(action_factors):
                # for non main factors add zero as padding element at position zero
                if position > 0 and len(group_vocabulary[position]) == 0:
                    group_vocabulary[position].append(pad_symbol)
                # get index or add new symbol
                if factor in group_vocabulary[position]:
                    factor_index = group_vocabulary[position].index(factor)
                else:
                    group_vocabulary[position].append(factor)
                    factor_index = len(group_vocabulary[position]) - 1
                group_index_map[position][index] = factor_index

        # use pad to signal to all vocabularies with pos > 0 that elements
        # shoudl be masked.
        full_to_factor_map = {}
        for position in group_index_map.keys():
            full_to_factor_map[position] = torch.zeros(len(vocabulary.symbols))
            for index in group_index_map[position]:
                full_to_factor_map[position][index] = \
                    group_index_map[position][index]

        # self._sanity_check(vocabulary, full_to_factor_map, group_index_map,
        #                   group_vocabulary)

        return full_to_factor_map, group_vocabulary

    def _get_action_factors(self, action):
        if '(' in action:
            action_base = action.split('(')[0]
            action_labels = action.split('(')[1][:-1].split(',')
            # add symbol to senses to avoid colliding with other numbers
            if action_base.startswith('COPY'):
                action_labels = [
                    f'_{a}' if re.match('[0-9]+', a) else a
                    for a in action_labels
                ]
            action_list = [action_base] + action_labels
        else:
            action_list = [action]
        return action_list

    def _sanity_check(self, vocabulary, full_to_factor_map, group_index_map,
                      group_vocabulary):

        # Sanity check: we can recover the original vocabulary
        for index, action in enumerate(vocabulary.symbols):
            action_factors = self._get_action_factors(action)
            rec_action_factors = []
            for position in group_vocabulary.keys():
                if position > 0 and full_to_factor_map[position][index] == 0:
                    break
                idx = group_index_map[position][index]
                rec_action_factors.append(group_vocabulary[position][idx])

            if rec_action_factors != action_factors:
                import ipdb; ipdb.set_trace(context=30)
                print()
