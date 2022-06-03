import collections
import copy
import itertools
import os

from amr_utils import convert_amr_to_tree, compute_pairwise_distance, get_node_ids
from transition_amr_parser.io import read_amr

import numpy as np
import torch
from tqdm import tqdm


def fake_align(amr):
    return {k: [0] for k in amr.nodes.keys()}


def format_value(val):
    if isinstance(val, float):
        return '{:.3f}'.format(val)
    return val


class Metric(object):
    _name = None

    @property
    def name(self):
        return self._name

    def __init__(self):
        self.state = collections.defaultdict(list)

    def update(self, gold, pred):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def print(self, result):
        header = '{}'.format(self.name)
        underl = '-' * len(header)
        output = '{}\n{}\n'.format(header, underl)
        for k, v in result.items():
            output += '- {} = {}\n'.format(k, format_value(v))
        print(output)


class MAPImpliedDistance(Metric):
    _name = 'MAP Implied Distance'

    def update(self, gold, pred):
        tree = convert_amr_to_tree(gold)
        pairwise_dist = compute_pairwise_distance(tree)
        node_ids = tree['node_ids']
        n = len(node_ids)

        def helper(amr, check_amr, max_amr_dist=0):
            vecs = collections.defaultdict(list)

            # TODO: Adjust for spans.
            # TODO: Double check bias between gold and pred, since gold has less alignments.
            for i in range(n):
                if node_ids[i] not in check_amr.alignments:
                    continue

                k = amr.alignments[node_ids[i]][0] - 1

                for j in range(i + 1, n):

                    # Only include aligned nodes.
                    if node_ids[j] not in check_amr.alignments:
                        continue

                    l = amr.alignments[node_ids[j]][0] - 1

                    # TODO: Support different views.
                    # if max_amr_dist > 0 and amr_dist > max_amr_dist:
                    #     continue

                    vecs['i'].append(i)
                    vecs['j'].append(j)
                    vecs['k'].append(k)
                    vecs['l'].append(l)

                    # amr_dist = pairwise_dist[i, j].item()
                    # tok_dist = np.abs(k - l).item()
                    # sq_diff = np.power(tok_dist - amr_dist, 2)
                    # res.append(sq_diff)

            if len(vecs) == 0:
                return None

            for k in list(vecs.keys()):
                vecs[k] = torch.tensor(vecs[k], dtype=torch.long)

            amr_dist = pairwise_dist[vecs['i'], vecs['j']]
            tok_dist = torch.abs(vecs['k'] - vecs['l'])
            sq_diff = torch.pow(tok_dist - amr_dist, 2)

            return sq_diff

        gold_res = helper(gold, gold)
        if gold_res is None:
            return
        pred_res = helper(pred, gold)
        if pred_res is None:
            return

        self.state['gold'].append(gold_res.float().mean().view(1))
        self.state['pred'].append(pred_res.float().mean().view(1))

    def finish(self):
        result = dict()
        result['pred'] = torch.cat(self.state['pred']).float().mean().item()
        result['gold'] = torch.cat(self.state['gold']).float().mean().item()

        diff = torch.cat(self.state['gold']) - torch.cat(self.state['pred'])
        index = torch.argsort(diff)
        show = 100
        print(' '.join([str(x) for x in index[:show].tolist()]))

        return result


class SentenceRecall(Metric):
    _name = 'Sentence Recall'

    def update(self, gold, pred):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        for node_id in gold_align.keys():
            p = pred_align[node_id]
            g = gold_align[node_id]

            total += 1

            if p == g:
                correct += 1

        # Don't count examples without alignments.
        if total == 0:
            return


        recall = correct / total

        self.state['recall'].append(recall)

    def finish(self):
        result = dict(recall=np.mean(self.state['recall']).item())
        return result


class CorpusRecall(Metric):
    _name = 'Corpus Recall'

    def update(self, gold, pred):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        for node_id in gold_align.keys():
            # ignore unaligned nodes for the purpose of computing stats
            if gold_align[node_id] is None:
                continue
            total += 1

            if pred_align[node_id][0] == gold_align[node_id][0]:
                correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)

    def finish(self):
        total = np.sum(self.state['total']).item()
        correct = np.sum(self.state['correct']).item()
        if total:
            recall = correct / total
        else:
            recall = 0
        result = collections.OrderedDict()
        result['correct'] = correct
        result['total'] = total
        result['recall'] = recall

        return result


class CorpusRecall_ExcludeNode(CorpusRecall):
    def __init__(self, exclude=['country', 'name', 'person']):
        super().__init__()

        self.exclude = exclude

    @property
    def name(self):
        return '{} excluding nodes = {}'.format(self._name, ', '.join(["'{}'".format(x) for x in self.exclude]))

    def update(self, gold, pred):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct, skipped = 0, 0, 0

        for node_id in gold_align.keys():
            node_name = gold.nodes[node_id]
            if node_name in self.exclude:
                skipped += 1
                continue

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            p = pred_align[node_id][0] - 1
            g = gold_align[node_id][0] - 1

            total += 1

            if p == g:
                correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)
        self.state['skipped'].append(skipped)

    def finish(self):
        result = super().finish()
        result['skipped'] = np.sum(self.state['skipped']).item()
        return result


class CorpusRecall_DuplicateText(CorpusRecall):

    @property
    def name(self):
        return '{} with fixed duplicate text'.format(self._name)

    def update(self, gold, pred):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0
        duplicate = 0
        fixed = 0

        text_counts = collections.Counter(gold.tokens)

        for node_id in gold_align.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            p = pred_align[node_id][0] - 1
            g = gold_align[node_id][0] - 1

            text_name = gold.tokens[g]
            if text_counts[text_name] > 1:
                duplicate += 1

                if p != g:
                    fixed += 1
                    p = g

            total += 1

            if p == g:
                correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)
        self.state['duplicate'].append(duplicate)
        self.state['fixed'].append(fixed)

    def finish(self):
        result = super().finish()
        result['duplicate'] = np.sum(self.state['duplicate']).item()
        result['fixed'] = np.sum(self.state['fixed']).item()
        return result


class CorpusRecall_IgnoreURL(CorpusRecall):

    @property
    def name(self):
        return '{} ignoring URLs'.format(self._name)

    def update(self, gold, pred):
        sentence = ' '.join(gold.tokens)

        if 'http' in sentence:
            self.state['skipped'].append(1)
            return

        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        for node_id in gold_align.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            p = pred_align[node_id][0] - 1
            g = gold_align[node_id][0] - 1

            total += 1

            if p == g:
                correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)

    def finish(self):
        result = super().finish()
        result['skipped'] = np.sum(self.state['skipped']).item()
        return result


class CorpusRecall_ForCOFILL(CorpusRecall):

    @property
    def name(self):
        return 'Node-to-Token F1'

    def update(self, gold, pred, only_1=False):
        gold_align, pred_align = gold.alignments, pred.alignments

        m = collections.Counter()

        for node_id in gold.nodes.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                if node_id not in pred_align or pred_align[node_id] is None:
                    pass

                else:
                    p0 = pred_align[node_id][0]
                    p1 = pred_align[node_id][-1]
                    pset = set(range(p0, p1 + 1))
                    m['fp'] += len(pset)

                continue

            g0 = gold_align[node_id][0]
            g1 = gold_align[node_id][-1]
            gset = set(range(g0, g1 + 1))

            assert g0 >= 0
            assert g1 >= g0

            # Penalty for not predicting.
            if node_id not in pred_align or pred_align[node_id] is None:
                m['fn'] += len(gset)
                continue

            p0 = pred_align[node_id][0]
            p1 = pred_align[node_id][-1]
            pset = set(range(p0, p1 + 1))

            m['tp'] += len(set.intersection(pset, gset))
            m['fn'] += len(gset) - len(set.intersection(pset, gset))
            m['fp'] += len(pset) - len(set.intersection(pset, gset))

        for k, v in m.items():
            self.state[k].append(v)

    def finish(self):
        tp = np.sum(self.state['tp']).item()
        fn = np.sum(self.state['fn']).item()
        fp = np.sum(self.state['fp']).item()

        total = (tp + fn)

        rec = tp / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        result = collections.OrderedDict()
        result['prec'] = prec
        result['rec'] = rec
        result['f1'] = f1

        result['total'] = total

        return result


class CorpusRecall_ForCOFILL_EasySpan(CorpusRecall_ForCOFILL):

    @property
    def name(self):
        return 'Node-to-Token F1 (easy span)'

    def update(self, gold, pred, only_1=False):
        gold_align, pred_align = gold.alignments, pred.alignments

        m = collections.Counter()

        for node_id in gold.nodes.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                if node_id not in pred_align or pred_align[node_id] is None:
                    pass

                else:
                    p0 = pred_align[node_id][0]
                    p1 = pred_align[node_id][-1]
                    pset = set(range(p0, p1 + 1))
                    m['fp'] += len(pset)

                continue

            g0 = gold_align[node_id][0]
            g1 = gold_align[node_id][-1]
            gset = set(range(g0, g1 + 1))

            assert g0 >= 0
            assert g1 >= g0

            # Penalty for not predicting.
            if node_id not in pred_align or pred_align[node_id] is None:
                m['fn'] += len(gset)
                continue

            p0 = pred_align[node_id][0]
            p1 = pred_align[node_id][-1]
            pset = set(range(p0, p1 + 1))


            intersect = len(set.intersection(pset, gset))
            if intersect > 0:
                pset = set.union(gset, pset)

            m['tp'] += len(set.intersection(pset, gset))
            m['fn'] += len(gset) - len(set.intersection(pset, gset))
            m['fp'] += len(pset) - len(set.intersection(pset, gset))

        for k, v in m.items():
            self.state[k].append(v)


class CorpusRecall_WithGoldSpans(CorpusRecall):

    @property
    def name(self):
        return '{} using spans for gold'.format(self._name)

    def update(self, gold, pred, only_1=False):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        for node_id in gold_align.keys():

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            # Penalty for not predicting.
            if node_id not in pred_align or pred_align[node_id] is None:
                total += 1
                continue

            total += 1

            g0 = gold_align[node_id][0] - 1
            g1 = gold_align[node_id][-1] - 1

            if only_1 or len(pred_align[node_id]) == 1:
                assert len(pred_align[node_id]) == 1

                p = pred_align[node_id][0] - 1

                if (p >= g0) and (p <= g1):
                    correct += 1

            else:
                p0 = pred_align[node_id][0] - 1
                p1 = pred_align[node_id][-1] - 1

                gset = set(range(g0, g1 + 1))
                pset = set(range(p0, p1 + 1))

                if len(set.intersection(pset, gset)) > 0:
                    correct += 1

        self.state['total'].append(total)
        self.state['correct'].append(correct)


class CorpusRecall_WithDupsAndSpans(CorpusRecall):

    @property
    def name(self):
        return '{} with dups and spans'.format(self._name)

    def update(self, gold, pred):
        gold_align, pred_align = gold.alignments, pred.alignments
        total, correct = 0, 0

        text_counts = collections.Counter(gold.tokens)

        is_span = False
        is_duplicate = False
        is_fixed = False

        for node_id in gold_align.keys():
            assert len(pred_align[node_id]) == 1
            p = pred_align[node_id][0] - 1

            # Ignore unaligned nodes
            if gold_align[node_id] is None:
                continue

            g0 = gold_align[node_id][0] - 1
            g1 = gold_align[node_id][-1] - 1

            if len(gold_align[node_id]) > 0:
                is_span = True

            is_correct = False
            if (p >= g0) and (p <= g1):
                is_correct = True

            text_name = gold.tokens[g0]
            if text_counts[text_name] > 1:
                is_duplicate = True

                if not is_correct:
                    is_correct = True
                    is_fixed = True

            if is_correct:
                correct += 1

            total += 1

        is_duplicate_and_fixed = is_duplicate and is_fixed
        is_both = is_span and is_duplicate
        is_both_and_fixed = is_both and is_fixed

        def to_int(b):
            return 1 if b else 0

        self.state['total'].append(total)
        self.state['correct'].append(correct)
        self.state['is_duplicate'].append(to_int(is_duplicate))
        self.state['is_duplicate_and_fixed'].append(to_int(is_duplicate_and_fixed))
        self.state['is_span'].append(to_int(is_span))
        self.state['is_both'].append(to_int(is_both))
        self.state['is_both_and_fixed'].append(to_int(is_both_and_fixed))

    def finish(self):
        result = super().finish()
        for k in ['is_duplicate', 'is_duplicate_and_fixed', 'is_span', 'is_both', 'is_both_and_fixed']:
            result[k] = np.sum(self.state[k]).item()
        return result


class EvalAlignments(object):
    def run(self, path_gold, path_pred, verbose=True, only_MAP=False, flexible=False, subset=False, increment=False):

        assert os.path.exists(path_gold)
        assert os.path.exists(path_pred)

        if only_MAP:
            metrics = [
                MAPImpliedDistance(),
            ]
        else:
            metrics = [
                #SentenceRecall(),
                #CorpusRecall(),
                #CorpusRecall_ExcludeNode(),
                #CorpusRecall_DuplicateText(),
                #CorpusRecall_IgnoreURL(),
                CorpusRecall_WithGoldSpans(),
                CorpusRecall_ForCOFILL(),
                CorpusRecall_ForCOFILL_EasySpan(),
                #CorpusRecall_WithDupsAndSpans(),
            ]

        gold = read_amr(path_gold)
        pred = read_amr(path_pred)
        print(f'N = {len(gold)} {len(pred)}')

        jamr = False
        remove_wiki_gold = True
        remove_wiki_pred = True
        attempt_rotate = True
        count_tokenization = True
        restrict_tokenization = False

        if remove_wiki_pred:
            for amr in pred:
                new_edges = []

                for node_id, edge_label, node_id_out in list(amr.edges):
                    if edge_label == ':wiki':
                        if node_id_out in amr.nodes:
                            del amr.nodes[node_id_out]
                        if node_id_out in amr.alignments:
                            del amr.alignments[node_id_out]

                for node_id, edge_label, node_id_out in list(amr.edges):
                    if node_id not in amr.nodes:
                        continue
                    if node_id_out not in amr.nodes:
                        continue
                    new_edges.append((node_id, edge_label, node_id_out))

                amr.edges = new_edges

        if remove_wiki_gold:
            for amr in gold:
                new_edges = []

                for node_id, edge_label, node_id_out in list(amr.edges):
                    if edge_label == ':wiki':
                        if node_id_out in amr.nodes:
                            del amr.nodes[node_id_out]
                        if node_id_out in amr.alignments:
                            del amr.alignments[node_id_out]

                for node_id, edge_label, node_id_out in list(amr.edges):
                    if node_id not in amr.nodes:
                        continue
                    if node_id_out not in amr.nodes:
                        continue
                    new_edges.append((node_id, edge_label, node_id_out))

                amr.edges = new_edges

        if jamr:
            for amr in pred:
                amr.alignments = amr.jamr_alignments

        if increment:
            def increment_node_id(node_id):
                return '.'.join([str(int(x) + 1) for x in node_id.split('.')])
            for amr in pred:
                new_alignments = {}
                new_nodes = {}
                new_edges = []

                for node_id, node_name in amr.nodes.items():
                    new_node_id = increment_node_id(node_id)

                    new_nodes[new_node_id] = node_name
                    if node_id in amr.alignments:
                        new_alignments[new_node_id] = amr.alignments[node_id]

                for a, b, c in amr.edges:
                    new_edges.append((increment_node_id(a), b, increment_node_id(c)))

                amr.alignments = new_alignments
                amr.nodes = new_nodes
                amr.edges = new_edges
                amr.root = increment_node_id(amr.root)

        if subset:
            d_gold = {amr.id: amr for amr in gold}
            d_pred = {amr.id: amr for amr in pred}
            keys = list(set.intersection(set(d_gold.keys()), set(d_pred.keys())))

            gold = [d_gold[k] for k in keys]
            pred = [d_pred[k] for k in keys]
            print(f'N = {len(gold)} {len(pred)}')

        if attempt_rotate:
            # For all pred nodes, try to match structure of gold nodes.
            def rotate(g, p):
                def get_edges(x):
                    new_edges = collections.defaultdict(set)

                    for node_id, edge_label, node_id_out in x.edges:
                        assert node_id in x.nodes
                        assert node_id_out in x.nodes
                        if node_id.count('.') == node_id_out.count('.') - 1:
                            if node_id_out.startswith(node_id):
                                new_edges[node_id].add(node_id_out)

                    return new_edges

                g_edges = get_edges(g)
                p_edges = get_edges(p)

                def attempt(node_id, p_node_id, new_node_list):

                    assert g.nodes[node_id] == p.nodes[p_node_id]

                    children = list(sorted(g_edges[node_id]))
                    if len(children) == 0:
                        return new_node_list + [p_node_id]

                    c_possible = []
                    for c_node_id in children:
                        p_children = p_edges[p_node_id]
                        c_node_name = g.nodes[c_node_id]
                        possible = sorted([x for x in p_children if x not in new_node_list and p.nodes[x] == c_node_name])
                        if len(possible) == 0:
                            raise ValueError('not enough')
                        c_possible.append(possible)


                    perm_list = list(itertools.product(*c_possible))

                    # print(node_id, g.nodes[node_id], p_node_id, new_node_list, children, c_possible, perm_list)

                    next_node_list = None
                    for possible_perm in perm_list:
                        next_node_list = new_node_list + [p_node_id]

                        try:
                            for possible_node_id, c_node_id in zip(possible_perm, children):
                                next_node_list = attempt(c_node_id, possible_node_id, next_node_list)
                        except ValueError:
                            next_node_list = None
                            continue

                        break

                    if next_node_list is None:
                        raise ValueError('no res')

                    return next_node_list

                g_node_list = list(sorted(g.nodes.keys()))

                assert tuple(sorted(g.nodes.values())) == tuple(sorted(p.nodes.values()))
                assert g.nodes[g.root] == p.nodes[p.root]

                new_node_list = attempt(g.root, p.root, [])

                assert tuple(sorted(new_node_list)) == tuple(sorted(p.nodes.keys())), (new_node_list, p.nodes.keys())
                pred_to_gold = {p_node_id: g_node_id for g_node_id, p_node_id in zip(g_node_list, new_node_list)}

                p.nodes = {pred_to_gold[k]: v for k, v in p.nodes.items()}
                p.alignments = {pred_to_gold[k]: v for k, v in p.alignments.items()}
                p.edges = [(pred_to_gold[a], b, pred_to_gold[c]) for a, b, c in p.edges]
                p.root = pred_to_gold[p.root]

                return p

            new_corpus = []
            for g, p in zip(gold, pred):
                should_rotate = False
                for node_id, node_name in g.nodes.items():
                    if node_id not in p.nodes or node_name != p.nodes[node_id]:
                        should_rotate = True
                        break

                if should_rotate:
                    p = rotate(g, p)
                new_corpus.append(p)
            pred = new_corpus

        if count_tokenization:
            m = collections.Counter()
            for g, p in zip(gold, pred):
                if ' '.join(g.tokens) != ' '.join(p.tokens):
                    m['different'] += 1
                m['total'] += 1
            print('count_tokenization', m)

        if restrict_tokenization:
            new_g, new_p = [], []
            for g, p in zip(gold, pred):
                if ' '.join(g.tokens) != ' '.join(p.tokens):
                    continue
                new_g.append(g)
                new_p.append(p)
            print(f'restrict_tokenization {len(gold)} -> {len(new_g)}')
            gold = new_g
            pred = new_p

        # check node names
        for g, p in zip(gold, pred):
            for k, v in g.nodes.items():
                assert k in p.nodes, (k, g.nodes, p.nodes)
                assert v == p.nodes[k], (k, v, p.nodes[k], g.id)

        if not flexible:
            assert len(gold) == len(pred), \
                f'{path_gold} and {path_pred} differ in size'

            for i, (g, p) in tqdm(enumerate(zip(gold, pred)), desc='eval'):

                for m in metrics:
                    m.update(g, p)

        else:
            assert len(gold) == len(pred)

            stats = collections.Counter()

            d_gold = {}
            d_dup = {}

            for g, p in zip(gold, pred):
                k = ' '.join(g.tokens)
                if k in d_gold:
                    stats['debug-duplicate'] += 1
                if k in d_gold and k not in d_dup:
                    d_dup[k] = g
                    stats['debug-distinct-duplicate'] += 1
                d_gold[k] = g

            for g, p in zip(gold, pred):

                keys_g = tuple(sorted(g.nodes.keys()))
                keys_p = tuple(sorted(p.nodes.keys()))

                name_g = tuple([g.nodes[k] for k in sorted(p.nodes.keys())])
                name_p = tuple([p.nodes[k] for k in sorted(p.nodes.keys())])

                if name_g != name_p:
                    stats['skip-node_name-mismatch'] += 1
                    continue

                if keys_g != keys_p:
                    #print('g', keys_g)
                    #print('p', keys_p)
                    stats['skip-node_id-mismatch'] += 1
                    continue

                for i_m, m in enumerate(metrics):
                    m.update(g, p)

                stats['ok'] += 1

            print(stats)



        if verbose:
            # clear line
            print('')

        output = {}

        for m in metrics:
            result = m.finish()
            if verbose:
                m.print(result)
            output[m.name] = result

        return output
