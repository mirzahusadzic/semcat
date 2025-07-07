import re
from typing import Optional
from collections import Counter

import penman
from penman.models.amr import model, reifications

import networkx
from experiment import standardizer

ALIGNMENT_REGEX = re.compile('(~([a-zA-Z]\\.?)?\\d+(,\\d+)*$)')

def _get_verbose_graph(graph: penman.Graph) -> Optional[networkx.DiGraph]:
    new_graph = networkx.DiGraph()

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

t1 ='(c2 / cause-01 :ARG1 (o / obligate-01:prep-test (y / you):ARG2 (s / scare-01:ARG0 (b / boogeyman:mod (c / company:name (n / name:op1 "Diebold"):wiki "Diebold_Nixdorf")):ARG1 y):polarity - :time (a / anymore)))'
t2 ='(c2 / cause-01 :ARG1 (o / obligate-01:ARG1 (y / you):ARG2 (s / scare-01:ARG0 (b / boogeyman:mod (c / company:name (n / name:op1 "Diebold"):wiki "Diebold_Nixdorf")):ARG1 y):polarity - :time (a / anymore)))'

t1 = standardizer.to_standard_amr(t1)
t2 = standardizer.to_standard_amr(t2)

t1 = penman.loads(t1.strip(), model=model)[0]
t2 = penman.loads(t2.strip(), model=model)[0]

t1 = _get_verbose_graph(t1)
t2 = _get_verbose_graph(t2)

