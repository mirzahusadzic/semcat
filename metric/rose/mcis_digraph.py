import networkx as nx
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

class PartitioningMCISFinder:
    def __init__(
        self,
        AG: np.ndarray,
        AH: np.ndarray,
        connected: bool = False,
    ):
        self.AG = AG
        self.AH = AH
        self.connected = connected

    def refine_label_classes(
        self,
        label_classes: List[Tuple[List[int], List[int]]],
        v: int,
        w: int,
    ) -> List[Tuple[List[int], List[int]]]:
        new_label_classes = []
        for G_nodes, H_nodes in label_classes:
            out_labels_G = {}
            in_labels_G = {}
            out_labels_H = {}
            in_labels_H = {}
            for u in G_nodes:
                out_labels_G[u] = self.AG[v-1, u-1]
                in_labels_G[u] = self.AG[u-1, v-1]
            for u in H_nodes:
                out_labels_H[u] = self.AH[w-1, u-1]
                in_labels_H[u] = self.AH[u-1, w-1]
            for out_label in set(out_labels_G.values()) & set(out_labels_H.values()):
                if out_label != 0:
                    new_G_nodes = [u for u in G_nodes if out_labels_G[u] == out_label]
                    new_H_nodes = [u for u in H_nodes if out_labels_H[u] == out_label]
                    if new_G_nodes and new_H_nodes:
                        new_label_classes.append((new_G_nodes, new_H_nodes))
            for in_label in set(in_labels_G.values()) & set(in_labels_H.values()):
                if in_label != 0:
                    new_G_nodes = [u for u in G_nodes if in_labels_G[u] == in_label]
                    new_H_nodes = [u for u in H_nodes if in_labels_H[u] == in_label]
                    if new_G_nodes and new_H_nodes:
                        new_label_classes.append((new_G_nodes, new_H_nodes))
        return new_label_classes

    def select_label_class(
        self,
        label_classes: List[Tuple[List[int], List[int]]],
        assignment_count: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
        if self.connected and assignment_count > 0:
            candidates = [lc for lc in label_classes if any(self.AG[v-1, u-1] != 0 or self.AG[u-1, v-1] != 0 for u in lc[0] for v in assignments)]
        else:
            candidates = label_classes
        if not candidates:
            return None
        return min(candidates, key=lambda lc: max(len(lc[0]), len(lc[1])))

    def calculate_bound(
        self,
        label_classes: List[Tuple[List[int], List[int]]],
    ) -> int:
        return sum(min(len(G_nodes), len(H_nodes)) for G_nodes, H_nodes in label_classes)

    def search(
        self,
        label_classes: List[Tuple[List[int], List[int]]],
        assignments: Dict[int, int],
        target: int,
    ) -> Optional[Dict[int, int]]:
        if len(assignments) == target:
            return assignments
        if len(assignments) + self.calculate_bound(label_classes) < target:
            return None
        label_class = self.select_label_class(label_classes, len(assignments))
        if label_class is None:
            return None
        v = label_class[0].pop()
        H_nodes = label_class[1][:]
        for w in H_nodes:
            label_class[1][:] = [u for u in H_nodes if u != w]
            assignments[v] = w
            new_label_classes = self.refine_label_classes(label_classes, v, w)
            search_result = self.search(new_label_classes, assignments, target)
            if search_result is not None:
                return search_result
            del assignments[v]
        label_class[1][:] = H_nodes
        return self.search([lc for lc in label_classes if lc[0]], assignments, target)

    def find_common_subgraph(self, target: int) -> Optional[Dict[int, int]]:
        label_classes = [([i for i in range(1, self.AG.shape[0] + 1)], [i for i in range(1, self.AH.shape[0] + 1)])]
        return self.search(label_classes, {}, target)

def create_adjacency_matrices(G: nx.DiGraph, H: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
    n = G.number_of_nodes()
    m = H.number_of_nodes()
    AG = np.full((n, n), None)
    AH = np.full((m, m), None)
    node_index_G = {node: i for i, node in enumerate(G.nodes)}
    node_index_H = {node: i for i, node in enumerate(H.nodes)}
    for i, j in G.edges():
        AG[node_index_G[i], node_index_G[j]] = G.edges[i, j]['name'][0]
    for i, j in H.edges():
        AH[node_index_H[i], node_index_H[j]] = H.edges[i, j]['name'][0]
    return AG, AH

def mcsplit_digraph(
    G: nx.DiGraph,
    H: nx.DiGraph,
    connected: bool = False,
) -> Dict[any, any]:
    AG, AH = create_adjacency_matrices(G, H)
    return PartitioningMCISFinder(AG, AH, connected).find_common_subgraph(min(G.number_of_nodes(), H.number_of_nodes()))