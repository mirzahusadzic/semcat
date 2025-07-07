import ast
import networkx as nx
from nltk import Tree
import matplotlib.pyplot as plt

def const_tree_to_digraph(str_tree):
    tree = Tree.fromstring(str_tree)
    G = nx.DiGraph()
    i = 0
    id_to_node = {}
    for subtree in tree.subtrees():
        node_label = f'z{i}'
        G.add_node(node_label, role=subtree.label())
        id_to_node[id(subtree)] = node_label
        if subtree.height() == 2:
            G.nodes[node_label]['name'] = subtree[0]
        i += 1
    for subtree in tree.subtrees():
        if subtree != tree:
            parent = next(tree.subtrees(filter=lambda x: subtree in x))
            if id(parent) in id_to_node:
                G.add_edge(id_to_node[id(parent)], id_to_node[id(subtree)])
    return G

def dep_to_digraph(dephead: list, deplabel: list, const: str):
    G = nx.DiGraph()
    tree = Tree.fromstring(const)
    postags_tokens = [(subtree[0], subtree.label()) for subtree in tree.subtrees() if subtree.height() == 2]
    for i in range(len(dephead)):
        if dephead[i] != 0:
            G.add_edge(f"z{dephead[i] - 1}", f"z{i}", name=deplabel[i])
            G.nodes[f"z{i}"]['name'] = postags_tokens[i][0]
            G.nodes[f"z{i}"]['pos'] = postags_tokens[i][1]
            G.nodes[f"z{i}"]['is_variable'] = True
        else:
            if G.number_of_edges() == 0:
                G.add_node(f"z{i}")
            G.nodes[f"z{i}"]['name'] = postags_tokens[i][0]
            G.nodes[f"z{i}"]['pos'] = postags_tokens[i][1]
            G.nodes[f"z{i}"]['is_variable'] = True
            G.nodes[f"z{i}"]['root'] = True
    return G

def export_trees(file_num: int, dataset: str, path: str):
    graphs = []
    for i in range(file_num):
        head = path + '/' + f'{dataset}_syndephead_{i}.txt'
        label = path + '/' + f'{dataset}_syndeplabel_{i}.txt'
        const = path + '/' + f'{dataset}_synconst_{i}.txt'
        with open(head, 'r') as f1, open(label, 'r') as f2, open(const, 'r') as f3:
            for line1, line2, line3 in zip(f1, f2, f3):
                graphs.append(dep_to_digraph(ast.literal_eval(line1.strip()), ast.literal_eval(line2.strip()), line3.strip()))
    return graphs

# plt.figure(figsize=(25,6))

# Test code for constituency parsing output
# const_1 = """(S (NP (DT This)) (VP (VBD was) (NP (NP (DT a) (NN series)) (PP (IN of) (NP (JJ nested) (JJ angular) (NNS standards)))) (, ,) (SBAR (IN so) (DT that) (S (NP (NP (NNS measurements)) (PP (IN in) (NP (NN azimuth) (CC and) (NN elevation)))) (VP (MD could) (VP (VB be) (VP (VBN done) (ADVP (RB directly)) (PP (IN in) (NP (NP (JJ polar) (NNS coordinates)) (ADJP (VBP relative) (PP (TO to) (NP (DT the) (JJ ecliptic)))))))))))) (. .))"""
# const_2 = """(S (NP (DT This)) (VP (VBD was) (NP (NP (DT a) (NN series)) (PP (IN of) (NP (JJ nested) (JJ polar) (NNS scales)))) (, ,) (SBAR (IN so) (DT that) (S (NP (NP (NNS measurements)) (PP (IN in) (NP (NN azimuth) (CC and) (NN elevation)))) (VP (MD could) (VP (VB be) (VP (VBN performed) (ADVP (RB directly)) (PP (IN in) (NP (NP (JJ angular) (NNS coordinates)) (ADJP (VBP relative) (PP (TO to) (NP (DT the) (JJ ecliptic)))))))))))) (. .))"""

# G = const_tree_to_digraph(const_1)
#
# Test code for dep parsing output
# dephead = [4, 4, 4, 0, 4, 8, 8, 5, 4, 19, 19, 19, 12, 13, 14, 14, 19, 19, 4, 19, 19, 23, 21, 23, 24, 27, 25, 4]
# deplabel = ['nsubj', 'cop', 'det', 'root', 'prep', 'amod', 'amod', 'pobj', 'punct', 'mark', 'mark', 'nsubjpass', 'prep', 'pobj', 'cc', 'conj', 'aux', 'auxpass', 'advcl', 'advmod', 'prep', 'amod', 'pobj', 'amod', 'prep', 'det', 'pobj', 'punct']
# postags = [('This', 'DT'), ('was', 'VBD'), ('a', 'DT'), ('series', 'NN'), ('of', 'IN'), ('nested', 'JJ'), ('angular', 'JJ'), ('standards', 'NNS'), (',', ','), ('so', 'IN'), ('that', 'DT'), ('measurements', 'NNS'), ('in', 'IN'), ('azimuth', 'NN'), ('and', 'CC'), ('elevation', 'NN'), ('could', 'MD'), ('be', 'VB'), ('done', 'VBN'), ('directly', 'RB'), ('in', 'IN'), ('polar', 'JJ'), ('coordinates', 'NNS'), ('relative', 'VBP'), ('to', 'TO'), ('the', 'DT'), ('ecliptic', 'JJ'), ('.', '.')]
# G = dep_to_digraph(dephead, deplabel, const_1)

# draw
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(G, pos)
# edge_labels = nx.get_edge_attributes(G, 'label')
# nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G, pos,
#                        arrowsize=20,
#                        arrowstyle='-|>',
#                        connectionstyle='arc3')
#
# plt.savefig('test.png')

# graphs = export_trees(9, 'paws1', '../../DP_para_results/PAWS')
