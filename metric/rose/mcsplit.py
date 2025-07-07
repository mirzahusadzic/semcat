# Custom Implement of mcsplit algorithm in IJCAI-2017 for AMR
import re

import numpy as np
import penman
import networkx as nx
from collections import Counter, defaultdict
from penman.models import amr
from penman.transform import dereify_edges

from typing import Optional, Tuple

ALIGNMENT_REGEX = re.compile('(~([a-zA-Z]\\.?)?\\d+(,\\d+)*$)')
REIFIED_CONCEPTS = {c for _, c, _, _ in amr.reifications}

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

    return penman.decode(penman.encode(penman.Graph(triples, top=top, metadata=g.metadata), model=amr.model),
                         model=amr.model)
def _relabeling_node(g: penman.Graph):
    return dereify_edges(_remove_duplicates(g), amr.model)

def _get_verbose_graph(graph: penman.Graph) -> Optional[nx.DiGraph]:
    graph = _relabeling_node(graph)
    new_graph = nx.DiGraph()

    for identifier in sorted(graph.instances(), key=lambda t: t.target or ''):
        if identifier.source is None:
            continue
        new_graph.add_node(identifier.source, name=identifier.target or '', is_variable=True)

    edges = {}
    attr_counts = Counter()
    for edge in sorted(graph.attributes(), key=lambda t: (t.source, t.role, t.target)):
        if edge.source is None or edge.target is None or (edge.role.lower() == ':wiki'):
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
    return new_graph

def search(future, M, incumbent, G, H, AG, AH):
    if len(M) > len(incumbent):
        incumbent = M
    bound = len(M) + sum(min(len(G0), len(H0)) for G0, H0 in future)
    if bound <= len(incumbent):
        return incumbent

    G_nodes, H_nodes = min(future, key=lambda gh: max(len(gh[0]), len(gh[1]))) # select label class
    v = max(G_nodes, key=lambda v: G.degree(v)) # select vertex
    edge_labels = set(AG.values()) | set(AH.values())

    for w in H_nodes:
        future_prime = set()
        for G_prime, H_prime in future:
            for l in edge_labels:
                G_pp = frozenset(u for u in G_prime if u != v and AG[(v, u)] == l)
                H_pp = frozenset(u for u in H_prime if u != w and AH[(w, u)] == l)
                if G_pp and H_pp:
                    future_prime.add((G_pp, H_pp))
        incumbent = search(future_prime, M | {(v, w)}, incumbent, G, H, AG, AH)

    G_prime = G_nodes - {v}
    future = future - {(G_nodes, H_nodes)}
    if G_prime:
        future.add((G_prime, H_nodes))
    return search(future, M, incumbent, G, H, AG, AH)

def to_adjacency_matrix(G):
    matrix = defaultdict(lambda: ('', ''))
    for u, v in G.edges:
        forward_label = ','.join(G[u][v].get('forward', []))
        backward_label = ','.join(G[v][u].get('forward', []))
        matrix[(u, v)] = (forward_label, backward_label)
        matrix[(v, u)] = (backward_label, forward_label)
    return matrix

def mcsplit(G: nx.DiGraph, H: nx.DiGraph):
    AG = to_adjacency_matrix(G)
    AH = to_adjacency_matrix(H)
    vertex_labels = set(G.nodes) & set(H.nodes) # consider labels
    future = {(frozenset(u for u in G.nodes if u in vertex_labels),
               frozenset(u for u in H.nodes if u in vertex_labels))}
    return search(future, set(), set(), G, H, AG, AH)
#
# Test code
def penman_to_nx(penman_string):
    data = penman.loads(penman_string, model=amr.model)
    return _get_verbose_graph(data[0])

msrp1= """(z0 / contrast-01:ARG2 (z1 / uphold-01:ARG0 (z2 / panel:part-of (z3 / government-organization:wiki "United_States_Court_of_Appeals_for_the_Third_Circuit":name (z4 / name:op1 "U.S.":op2 "Court":op3 "of":op4 "Appeals":op5 "for":op6 "the":op7 "3rd":op8 "Circuit"))):ARG1 (z5 / government-organization:ARG0-of (z6 / govern-01)):manner (z7 / vote-01:ARG0 z2:ARG2 (z8 / score-entity:op1 2:op2 1)):topic (z9 / case-03:location (z10 / state:wiki "New_Jersey":name (z11 / name:op1 "New":op2 "Jersey")))))"""
msrp2 = """(z0 / contrast-01:ARG1 (z1 / agree-01:ARG0 (z2 / person:ARG0-of (z3 / have-org-role-91:ARG3 (z4 / judge-01)):mod (z5 / federal):location (z6 / state:wiki "New_Jersey":name (z7 / name:op1 "New":op2 "Jersey")))):ARG2 (z8 / disagree-01:ARG0 (z9 / panel:part-of (z10 / government-organization:wiki "United_States_Court_of_Appeals_for_the_Third_Circuit":name (z11 / name:op1 "U.S.":op2 "Court":op3 "of":op4 "Appeals":op5 "for":op6 "the":op7 "3rd":op8 "Circuit")))))"""
# paws1 = """(z0 / spend-02:ARG0 (z1 / she):ARG1 (z2 / and:op1 (z3 / half:part-of (z4 / summer)):op2 (z5 / play-01:ARG0 z1:ARG3 (z6 / team:poss z1:mod (z7 / state:wiki "Alaska":name (z8 / name:op1 "Alaska")))):op3 (z9 / live-01:ARG0 z1:location (z10 / state:wiki "Oregon":name (z11 / name:op1 "Oregon")))):ARG2 (z12 / and:op1 (z13 / half:part-of z4):op2 (z14 / half:part-of z4)):time (z15 / and:op1 (z16 / summer:mod (z17 / sophomore):poss z1):op2 (z18 / summer:mod (z19 / junior):poss z1):op3 (z20 / summer:mod (z21 / senior):poss z1)))"""
# paws2 = """(z0 / spend-02:ARG0 (z1 / she):ARG1 (z2 / and:op1 (z3 / half:ARG1-of (z4 / include-91:ARG2 (z5 / summer:poss z1))):op2 (z6 / play-01:ARG0 z1:location (z7 / state:wiki "Oregon":name (z8 / name:op1 "Oregon"))):op3 (z9 / live-01:ARG0 z1:location z7)):accompanier (z10 / team:poss z1:mod (z11 / state:wiki "Alaska":name (z12 / name:op1 "Alaska"))):time (z13 / and:op1 (z14 / summer:ord (z15 / ordinal-entity:value 2):poss z1):op2 (z16 / summer:ord (z17 / ordinal-entity:value 2):poss z1):op3 (z18 / summer:ord (z19 / ordinal-entity:value 2):poss z1):op4 (z20 / summer:ord (z21 / ordinal-entity:value 2):poss z1):op5 (z22 / summer:ord (z23 / ordinal-entity:value 2):poss z1):op6 (z24 / summer:ord (z25 / ordinal-entity:value 2):poss z1):op7 (z26 / summer:ord (z27 / ordinal-entity:value 2):poss z1):op8 (z28 / summer:ord (z29 / ordinal-entity:value 2):poss z1):op9 (z30 / summer:ord (z31 / ordinal-entity:value 2):poss z1):op10 (z32 / summer:ord (z33 / ordinal-entity:value 3):poss z1):op11 (z34 / summer:ord (z35 / ordinal-entity:value 2):poss z1):op12 (z36 / summer:ord (z37 / ordinal-entity:value 3):poss z1)))"""
# penman_string1 = """(z0 / series:consist-of (z1 / standard:mod (z2 / angle-quantity:quant 1:scale (z3 / celsius)):ARG1-of (z4 / nest-01)):domain (z5 / this):purpose (z6 / possible-01:ARG1 (z7 / measure-01:ARG1 (z8 / and:op1 (z9 / azimuth):op2 (z10 / elevation)):manner (z11 / coordinate-01:ARG2 (z12 / polar):ARG1-of (z13 / relative-05:ARG2 (z14 / ecliptic)):ARG1-of (z15 / direct-02)))))"""
# penman_string2 = """(z0 / series:consist-of (z1 / scale:mod (z2 / pole):ARG1-of (z3 / nest-01)):domain (z4 / this):purpose (z5 / possible-01:ARG1 (z6 / perform-02:ARG1 (z7 / measure-01:ARG1 (z8 / and:op1 (z9 / azimuth):op2 (z10 / elevation))):manner (z11 / coordinate-01:ARG1 z8:ARG2 (z12 / relative-position:op1 (z13 / ecliptic)):mod (z14 / angle-quantity:quant 1:scale (z15 / celsius)):ARG1-of (z16 / direct-02)))))"""
G = penman_to_nx(msrp1)
H = penman_to_nx(msrp2)
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(35,15))
#
# pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(H, pos, with_labels=True, arrows=True)
#
# plt.savefig('amr_test.png')

# Find a maximum common induced subgraph
# mcsplit_result = mcsplit(G,H)
# print(mcsplit_result, len(mcsplit_result))
