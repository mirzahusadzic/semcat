import logging
import re
from collections import defaultdict, namedtuple, Counter
from typing import List, Optional
from itertools import permutations
from nltk import Tree

import networkx
import penman
import penman.model
import scipy
import scipy.stats
from penman import encode
from penman.models import amr
from penman.transform import dereify_edges

from .dp_nxgraph import export_trees

ALIGNMENT_REGEX = re.compile('(~([a-zA-Z]\\.?)?\\d+(,\\d+)*$)')
REIFIED_CONCEPTS = {c for _, c, _, _ in amr.reifications}
ONE_DICT = defaultdict(lambda: 1.0)
ZERO_DICT = defaultdict(lambda: 0.0)


NodeInfo = namedtuple('NodeInfo', ('name', 'neighbors'))
EdgeInfo = namedtuple('EdgeInfo', ('src', 'role', 'tgt', 'is_attribute'))


def _remove_duplicates(g: penman.Graph):
    triples = sorted(set(g.triples))
    top = g.top
    variables = [i.source for i in g.instances() if i.target not in REIFIED_CONCEPTS]
    if len(variables) and top not in variables:
        # Take a new top variable, to avoid wrong reification
        top = variables[0]
    elif len(triples) == len(g.triples):
        # IF there is no change, just return the original graph
        return g

    return penman.decode(encode(penman.Graph(triples, top=top, metadata=g.metadata), model=amr.model),
                         model=amr.model)


def _f1_score_new(set1, set2):
    truth_paths = set(set1)
    pred_paths = set(set2)
    true_pos_paths = truth_paths.intersection(pred_paths)

    return scipy.stats.hmean([len(true_pos_paths) / len(truth_paths) if truth_paths else float(len(truth_paths) == 0),
                              len(true_pos_paths) / len(pred_paths) if pred_paths else float(len(truth_paths) == 0)])


def _recall(set1, set2):
    truth_paths = set(set1)
    pred_paths = set(set2)
    true_pos_paths = truth_paths.intersection(pred_paths)

    return len(true_pos_paths) / len(truth_paths) if truth_paths else float(len(truth_paths) == 0)


def _init_weisfeiler_labels(graph: networkx.DiGraph) -> dict:
    return {n: ('var:' if d['is_variable'] else 'attr:') + d['name']
            for n, d in graph.nodes(data=True)}


def _neighborhood_aggregate(graph: networkx.DiGraph, node: str, node_labels: dict) -> set:
    edges = set()
    for nbr in graph.neighbors(node):
        edges.update([(r, node_labels[nbr]) for r in graph[node][nbr]['name']])
    return edges


def _get_connected_components(nodes1, nodes2, edges):
    if len(edges) == len(nodes1) * len(nodes2):  # All possible connections
        return [(nodes1, nodes2)]

    if len(edges) == 0:  # No connections (we need to separate them)
        return [(nodes1, []), ([], nodes2)]

    graph = networkx.Graph()
    graph.add_nodes_from([f'0.{a}' for a in nodes1])  # To distinguish the nodes
    graph.add_nodes_from([f'1.{a}' for a in nodes2])
    graph.add_edges_from([(f'0.{a}', f'1.{b}') for a, b in edges])

    groups = []
    for c in networkx.connected_components(graph):
        group_1 = []
        group_2 = []
        for n in c:
            if n.startswith('0.'):
                group_1.append(n[2:])
            else:
                group_2.append(n[2:])

        groups.append((group_1, group_2))

    return groups


def _enumerate_all_matching(label_to_nodes_1, label_to_nodes_2):
    # Update common node labels
    common_labels = set(label_to_nodes_1.keys()).intersection(label_to_nodes_2.keys())
    matchings = [[]]

    for l in common_labels:
        # Build permutations
        if len(label_to_nodes_1[l]) > len(label_to_nodes_2[l]):
            extension = [[tuple(t) for t in zip(p, label_to_nodes_2[l])]
                         for p in permutations(label_to_nodes_1[l], r=len(label_to_nodes_2[l]))]
        else:
            extension = [[tuple(t) for t in zip(label_to_nodes_1[l], p)]
                         for p in permutations(label_to_nodes_2[l], r=len(label_to_nodes_1[l]))]

        matchings = [m + e
                     for m in matchings
                     for e in extension]

    return matchings


class RoSE:
    PENMAN_MODEL = amr.model

    def __init__(self, ignore_wiki_role: bool = True, verbosity=None, num_iterations: int = 3,
                 precision_digit: int = 6, similarity_threshold_tau: float = 0.99, mode = 'AMR'):
        self.ignore_wiki_role = ignore_wiki_role
        self._prec = precision_digit
        self._n_iter = num_iterations
        self._tau = similarity_threshold_tau
        self._mode = mode

        assert 0.5 < self._tau < 1, 'similarity threshold value should in range (0.5, 1)'
        assert self._n_iter > 1, 'the number of iterations should be larger than 1'

        if verbosity is not None:
            logger = logging.getLogger(self.name())
            logger.setLevel(verbosity)
            self._debug = logger.debug
            self._info = logger.info
            self._warn = logger.warning
            self._is_logging = True
        else:
            self._debug = self._info = self._warn = lambda *args, **kwargs: ()
            self._is_logging = False

    def name(self):
        return self.__class__.__name__[:-6] + str(self._n_iter)  # Remove scorer

    def _match_nodes(self, graph1: networkx.DiGraph, graph2: networkx.DiGraph):
        # Collect all node labels
        node_labels_1 = _init_weisfeiler_labels(graph1)
        node_labels_2 = _init_weisfeiler_labels(graph2)

        def label_to_nodes(label: dict) -> dict:
            label_to_node = defaultdict(list)
            for n, l in label.items():
                label_to_node[l].append(n)

            return label_to_node

        def weisfeiler_leman_step(graph: networkx.DiGraph, labels: dict):
            new_connections = {}
            for node in graph.nodes:
                edges = _neighborhood_aggregate(graph, node, labels)
                new_connections[node] = edges

            return new_connections

        # Iterate!
        fixed_labels = set()
        for i in range(self._n_iter):
            label_to_nodes_1 = label_to_nodes(node_labels_1)
            label_to_nodes_2 = label_to_nodes(node_labels_2)

            # Update shared node labels
            shared_labels = set(label_to_nodes_1.keys()).intersection(label_to_nodes_2.keys())
            shared_labels.difference_update(fixed_labels)

            neighbors_1 = weisfeiler_leman_step(graph1, node_labels_1)
            neighbors_2 = weisfeiler_leman_step(graph2, node_labels_2)
            if self._is_logging:
                self._debug(f'Neighbors (REF) at iteration {i}: {neighbors_1}')
                self._debug(f'Neighbors (HYP) at iteration {i}: {neighbors_2}')

            # Find unique node label pair, assign it as equivalent pair
            new_node_labels_1 = {}
            new_node_labels_2 = {}
            for l in shared_labels:
                compatible_pairs = {(n1, n2)
                                    for n1 in label_to_nodes_1[l]
                                    for n2 in label_to_nodes_2[l]
                                    if _f1_score_new(neighbors_1[n1], neighbors_2[n2]) >= self._tau}

                if self._is_logging:
                    self._debug(f'Compatible pairs for label {l} in iteration {i}: {compatible_pairs}')

                # Make a group of names that can compatible to each other (using connected component algorithm)
                components = _get_connected_components(edges=compatible_pairs, nodes1=label_to_nodes_1[l],
                                                       nodes2=label_to_nodes_2[l])

                # Assign different names to make the system treat them as different things
                if len(components) == 1:
                    if len(components[0][0]) == len(components[0][1]) == 1:
                        fixed_labels.add(l)
                    continue

                for g, (g1, g2) in enumerate(components):
                    new_label = f'{l}#{g}'
                    new_node_labels_1.update({n: new_label for n in g1})
                    new_node_labels_2.update({n: new_label for n in g2})
                    if len(g1) == len(g2) == 1:
                        fixed_labels.add(new_label)

            # Update the changed node labels
            if new_node_labels_1 or new_node_labels_2:
                node_labels_1.update(new_node_labels_1)
                node_labels_2.update(new_node_labels_2)
                if self._is_logging:
                    self._debug(f'Label Update (REF): {new_node_labels_1}')
                    self._debug(f'Label Update (HYP): {new_node_labels_2}')
            else:
                # Early stopping if there is no change
                break

        # Make iteration according to common label lists
        label_to_nodes_1 = label_to_nodes(node_labels_1)
        label_to_nodes_2 = label_to_nodes(node_labels_2)
        if self._is_logging:
            self._debug(f'Labels (REF): {label_to_nodes_1}')
            self._debug(f'Labels (HYP): {label_to_nodes_2}')

        # Iterate over all possible matching
        for matching in _enumerate_all_matching(label_to_nodes_1, label_to_nodes_2):
            if self._is_logging:
                self._debug(f'Matching: {matching}')

            # Relabeling nodes by the matched result
            matching_for_graph1 = {}
            matching_for_graph2 = {}
            matched_nodes = []

            for i, (src, tgt) in enumerate(matching):
                new_node_var = f'matched_{i:03d}'
                matching_for_graph1[src] = new_node_var
                matching_for_graph2[tgt] = new_node_var
                matched_nodes.append(new_node_var)

            # Rename other variables to avoid uniting them
            for src, is_var in graph1.nodes(data='is_variable'):
                if is_var and src not in matching_for_graph1:
                    matching_for_graph1[src] = 'ref.' + src
            for tgt, is_var in graph2.nodes(data='is_variable'):
                if is_var and tgt not in matching_for_graph2:
                    matching_for_graph2[tgt] = 'hyp.' + tgt

            new_graph1 = networkx.relabel_nodes(graph1, matching_for_graph1, copy=True)
            new_graph2 = networkx.relabel_nodes(graph2, matching_for_graph2, copy=True)
            yield new_graph1, new_graph2, matched_nodes

    def _relabeling_node(self, g: penman.Graph):
        try:
            return dereify_edges(_remove_duplicates(g), RoSE.PENMAN_MODEL)
        except Exception as e:
            if self._is_logging:
                self._debug(f'Reification/Relabeling failed for the following graph: {encode(g)}', exc_info=e)
            raise e

    def _get_verbose_graph(self, graph: penman.Graph) -> Optional[networkx.DiGraph]:
        graph = self._relabeling_node(graph)
        new_graph = networkx.DiGraph()

        for identifier in sorted(graph.instances(), key=lambda t: t.target or ''):
            if identifier.source is None:
                continue
            new_graph.add_node(identifier.source, name=identifier.target or '', is_variable=True)

        edges = {}
        attr_counts = Counter()
        for edge in sorted(graph.attributes(), key=lambda t: (t.source, t.role, t.target)):
            if edge.source is None or edge.target is None or (self.ignore_wiki_role and edge.role.lower() == ':wiki'):
                continue

            target = ALIGNMENT_REGEX.sub('', edge.target) if type(edge.target) is str else str(edge.target)
            attr_counts[target] += 1
            attr_value = f'attr:{target}.{attr_counts[target]}'
            new_graph.add_node(attr_value, name=target, is_variable=False)

            role = ALIGNMENT_REGEX.sub('', edge.role)

            if (edge.source, attr_value) not in edges:
                edges[edge.source, attr_value] = set()
            edges[edge.source, attr_value].add(role)

        # Group edge roles by source and target
        for edge in sorted(graph.edges(), key=lambda t: (t.source, t.role, t.target)):
            if edge.source is None or edge.target is None:
                continue

            role = ALIGNMENT_REGEX.sub('', edge.role)

            if (edge.source, edge.target) not in edges:
                edges[edge.source, edge.target] = set()
            edges[edge.source, edge.target].add(role)

        for src, tgt in sorted(edges.keys()):
            if new_graph.has_edge(src, tgt):
                continue

            forward_roles = edges[src, tgt]
            backward_roles = edges.get((tgt, src), set())
            connections = len(forward_roles) + len(backward_roles)

            new_graph.add_edge(src, tgt,
                               name=sorted(forward_roles.union({r + '-of' for r in backward_roles})),
                               forward=forward_roles,
                               connections=connections)
            new_graph.add_edge(tgt, src,
                               name=sorted(backward_roles.union({r + '-of' for r in forward_roles})),
                               forward=backward_roles,
                               connections=connections)

        if self._is_logging:
            self._debug(f'Creating Graph: (V={new_graph.nodes}, E={new_graph.edges(data=True)})')
        return new_graph

    def _get_dp_graph(self, heads: list, labels: list, words: str):
        tree = Tree.fromstring(words)
        postags = [(subtree[0], subtree.label()) for subtree in tree.subtrees() if subtree.height() == 2]
        G = networkx.DiGraph()
        for i in range(len(heads)):
            if heads[i] != 0:
                G.add_edge(postags[heads[i] - 1][0], postags[i][0], label=labels[i])
        return G

    def compute_graph_similarity(self, reference: networkx.DiGraph, hypothesis: networkx.DiGraph) -> float:
        maximum_score = 0.0
        for reference, hypothesis, matched_names in self._match_nodes(reference, hypothesis):
            if self._is_logging:
                self._debug(f'Number of matched nodes: {len(matched_names)}')
            reference_edges = {(s, t)
                               for s, t in reference.edges()
                               }
            reference_edges.update({(n, '/', d['name'])
                                    for n, d in reference.nodes(data=True) if d['is_variable']})
            hypothesis_edges = {(s, t)
                                for s, t in hypothesis.edges()
                                }
            hypothesis_edges.update({(n, '/', d['name'])
                                     for n, d in hypothesis.nodes(data=True) if d['is_variable']})
            edge_true_positive = reference_edges.intersection(hypothesis_edges)

            truth = len(reference_edges)
            positive = len(hypothesis_edges)
            true_positive = len(edge_true_positive)

            pr_recall = true_positive / truth if truth else float(true_positive == 0)
            pr_precision = true_positive / positive if positive else float(true_positive == 0)
            score = scipy.stats.hmean([pr_precision, pr_recall])

            maximum_score = min(max(maximum_score, score), 1.0)  # For numerical stability of pagerank
            if maximum_score == 1.0:
                break

        return round(maximum_score, self._prec)

    def compute(self, references: List[networkx.DiGraph], hypotheses: List[networkx.DiGraph], return_all_scores: bool = False):
        assert len(references) == len(hypotheses), 'The length of references and hypotheses should be same!'
        assert len(references) > 0, 'List should not be empty!'

        assert isinstance(references[0], networkx.DiGraph), 'Reference should be a list of networkx.DiGraph!'
        assert isinstance(hypotheses[0], networkx.DiGraph), 'Hypotheses should be a list of networkx.DiGraph!'

        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_id = references.index(ref)
            hyp_id = hypotheses.index(hyp)

            # assert hyp_id == "NO_ID" or ref_id == hyp_id, \
            #     f'Metadata mismatch! Reference ID {ref_id} is given but we received hypothesis with ID {hyp_id}'

            try:
                score_current = self.compute_graph_similarity(ref, hyp)
            except Exception as e:
                self._warn(f'Error occurred when computing score of {ref_id}', exc_info=e)
                score_current = 0.0

            if self._is_logging:
                self._info(f'[#{len(scores)}/{ref_id}] Score for this item: {score_current}')
            scores.append(score_current)

        average = round(sum(scores, 0.0) / len(scores), self._prec)
        name = self.name()
        output_dict = {name: average}
        if return_all_scores:
            output_dict[f'list_{name}'] = scores

        return output_dict

    def compute_from_files(self, ref_files: List, hyp_files: List,
                           encoding: str = None, return_all_scores: bool = False) -> dict:
        if self._mode == 'DP':
            parsed_r = [i.as_posix() for i in ref_files]
            parsed_h = [i.as_posix() for i in hyp_files]
            references = export_trees(int(parsed_r[0]), parsed_r[1], parsed_r[2])
            hypotheses = export_trees(int(parsed_h[0]), parsed_h[1], parsed_h[2])
            return self.compute(references, hypotheses, return_all_scores)

        else:
            try:
                references = [graph
                              for f in ref_files
                              for graph in penman.load(f, model=model, encoding=encoding)]
                hypotheses = [graph
                              for f in hyp_files
                              for graph in penman.load(f, model=model, encoding=encoding)]

                return self.compute(references, hypotheses, return_all_scores)
            except penman.exceptions.DecodeError:
                return {self.name(): 0.0, f'list_{self.name()}': []}

    def compute_from_string(self, ref_string: str, hyp_string: str, model: penman.model.Model = None,
                            return_all_scores: bool = False) -> dict:
        if model is None:
            model = RoSE.PENMAN_MODEL

        try:
            references = penman.loads(ref_string, model=model)
            hypotheses = penman.loads(hyp_string, model=model)

            return self.compute(references, hypotheses, return_all_scores)
        except penman.exceptions.DecodeError:
            return {self.name(): 0.0, f'list_{self.name()}': []}
