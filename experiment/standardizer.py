from typing import Tuple, List
from collections import Counter, defaultdict

import penman
from penman.codec import format
from penman.layout import configure, rearrange, interpret, reconfigure
from penman.models.amr import model, reifications
from penman.transform import dereify_edges, reify_attributes

REIFIED_CONCEPTS = {c for _, c, _, _ in reifications}


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

    return penman.decode(penman.encode(penman.Graph(triples, top=top, metadata=g.metadata),
                                       model=model),
                         model=model)

def sort_key(tuple):
    return (tuple[0][0], len(tuple[1]), tuple[1], int(tuple[0][1:]))


def remove_all_duplicates(lst):
    seen = set()
    duplicates = set()
    result = []

    for item in lst:
        if item[1] in seen:
            duplicates.add(item[1])
        seen.add(item[1])

    for item in lst:
        if item[1] not in duplicates:
            result.append(item)

    return result


def top_tuples(tuples, graph):
    max_value = max(t[1] for t in tuples)
    max_tuples = [t[0] for t in tuples if t[1] == max_value]
    sorted_data = []
    while sorted_data is not []:
        final_candidates = []
        for s, r, _ in graph.triples:
            if len(max_tuples) == 1 and s == max_tuples[0]:
                final_candidates.append((s, _))
                break
            elif s in max_tuples and r == ':instance':
                final_candidates.append((s, _))
        sorted_data = sorted(final_candidates, key=sort_key)
        sorted_data = remove_all_duplicates(sorted_data)
        if sorted_data == [] and max_value > 1:
            max_value-=1
            max_tuples = [t[0] for t in tuples if t[1] == max_value]
            if max_value == 1 and max_tuples == []:
                sorted_data = final_candidates
                break
        else:
            break
    return sorted_data[0]


def canonicalize_graph(g: penman.Graph):
    triples = sorted(set(g.triples))
    new_triples = []
    for triple in triples:
        new_triples.append(model.deinvert(triple))
    triples = new_triples
    new_triples = []
    for triple in triples:
        new_triples.append(model.canonicalize(triple))
    out_counter = Counter([s for s, r, _ in new_triples if r != '/'])
    new_graph = penman.Graph(new_triples, metadata=g.metadata)
    top = top_tuples(out_counter.items(), new_graph)[0]
    return penman.decode(penman.encode(new_graph, top=top,
                          model=model), model=model)


def to_standard_amr(graph: str) -> str:
    graph = penman.loads(graph.strip(), model=model)[0]
    tree = reconfigure(graph, top=graph.top, model=model)
    rearrange(tree, key=model.canonical_order)
    tree.reset_variables('{prefix}{i}')
    graph = canonicalize_graph(dereify_edges(_remove_duplicates(penman.interpret(tree, model=model)), model))

    # Convert it into a tree and rearrange
    tree = reconfigure(graph, top=graph.top, model=model)
    rearrange(tree, key=model.canonical_order)
    tree.reset_variables('{prefix}{i}')
    return format(tree).strip()


# def get_connections(graph: penman.Graph) -> List[Tuple[str, str, set]]:
#     instances = {}
#     triples = defaultdict(list)
#     for s, r, t in graph.triples:
#         if r == '/':
#             instances[s] = t
#         else:
#             triples[s].append((r, t))
#             triples[t].append((r + '-of', s))
#
#     # Extend for 2-hop connections
#     new_triples = defaultdict(list)
#     for s in triples:
#         for r, t in triples[s]:
#             for r2, t2 in triples[t]:
#                 if t2 != s:
#                     new_triples[s].append((r + ' ' + r2, t2))
#
#     triples = {s: triples[s] + new_triples[s]
#                for s in triples}
#
#     connections = []
#     for i, name in instances.items():
#         tpl_i = triples.get(i, [])
#         tpl_i = [(r, instances.get(t, t))
#                  for r, t in tpl_i]
#         connections.append((name, i, set(tpl_i)))
#
#     return connections
#
#
# def permutation(variables: list) -> list:
#     if len(variables):
#         return [tuple()]
#
#     return [
#         (v,) + rest
#         for v in variables[0]
#         for rest in permutation(variables[1:])
#     ]
#
#
# def jaccard(*sets) -> float:
#     intersections = None
#     unions = None
#     for s in sets:
#         intersections = set(s) if intersections is None else intersections.intersection(s)
#         unions = set(s) if unions is None else unions.union(s)
#
#     return len(intersections) / len(unions)
#
# def to_standard_amr(*penmans: str) -> Tuple[str, ...]:
#     graphs = []
#     connections = []
#     name_counter = []
#     for graph in penmans:
#         graph = penman.loads(graph.strip(), model=model)[0]
#         graph = dereify_edges(_remove_duplicates(graph), model)
#         graphs.append(graph)
#
#         conn = get_connections(graph)
#         connections.append(conn)
#         name_counter.append(Counter([n for n, _, _ in conn]))
#
#     # Select a common node as top
#     occurrences = name_counter[0]
#     for counter in name_counter[1:]:
#         occurrences = occurrences & counter
#
#     occurrences = {k: v for k, v in occurrences.items() if v > 0}
#     top = [g.top for g in graphs]
#     if len(occurrences):
#         # Take the least appearing name
#         top_name = min(occurrences.keys(), key=lambda n: occurrences[n])
#         variable_candidates = [[(v, l) for n, v, l in conn if n == top_name] for conn in connections]
#         # Take the most similar pair
#         variable_pairs = max(permutation(variable_candidates), key=lambda p: jaccard(*[t for _, t in p]))
#         top = [v for v, _ in variable_pairs]
#
#     graphs = [penman.Graph(graph.triples, top=t, metadata=graph.metadata)
#               for graph, t in zip(graphs, top)]
#
#     # Convert them into trees
#     outputs = []
#     for graph in graphs:
#         tree = configure(graph, top=graph.top, model=model)
#         rearrange(tree, key=model.canonical_order)
#         tree.reset_variables('x{i:03d}')
#         outputs.append(format(tree).strip())
#
#     return tuple(outputs)
#
# r1 = '(p / person :ARG0-of (r / rape-01:ARG1 (c / child:ARG1-of (s / see-01:ARG0 (m / man:ARG0-of (d / do-02:ARG1 (n / nothing)))):ARG1-of (v / victimize-01:ARG0 m):mod (t / that))):ARG0-of (v2 / victimize-01:ARG1 c))'
# r2 = '(c / child :mod (t / that):ARG1-of (r / rape-01:ARG0 (p / person)):ARG1-of (s / see-01:ARG0 (m / man:ARG0-of (d / do-02:ARG1 (n / nothing)))):ARG1-of (v / victimize-01:ARG0 p):ARG1-of (v2 / victimize-01:ARG0 m))'
#
# s1 = '(h2 / have-rel-role-91 :ARG0 (p / person:ARG0-of (h / have-org-role-91:ARG1 (m / military:name (n / name:op1 "Coast":op2 "Guard"):wiki "United_States_Coast_Guard"):ARG2 (m2 / member))):ARG1 (i / i):ARG2 (d / dad))'
# s2 = '(h / have-org-role-91 :ARG0 (p / person:ARG0-of (h2 / have-rel-role-91:ARG1 (i / i):ARG2 (d / dad))):ARG1 (m / military:name (n / name:op1 "Coast":op2 "Guard"):wiki "United_States_Coast_Guard"):ARG2 (m2 / member))'
#
# paws 1694
# s1 = '(z0 / and :op1 (z1 / own-01:ARG0 (z2 / and:op1 (z3 / person:wiki -:name (z4 / name:op1 "Rick")):op2 (z5 / person:wiki - :name (z6 / name:op1 "Sheri":op2 "Dorritie"))):ARG1 (z7 / animal:wiki - :name (z8 / name:op1 "Megasaurus"))):op2 (z9 / own-01:ARG0 (z10 / person:wiki - :name (z11 / name:op1 "Mike":op2 "West":op3 "Transaurus")):ARG1 z7))'
# s2 = '(z0 / and :op1 (z1 / own-01:ARG0 (z2 / person:wiki -:name (z3 / name:op1 "Mike":op2 "West")):ARG1 (z4 / animal:wiki - :name (z5 / name:op1 "Megasaurus"))):op2 (z6 / own-01:ARG0 (z7 / and :op1 (z8 / person:wiki - :name (z9 / name:op1 "Rick")):op2 (z10 / person:wiki - :name (z11 / name:op1 "Sheri":op2 "Dorritie"))):ARG1 (z12 / animal:wiki - :name (z13 / name:op1 "Transaurus"))))'
#
# if __name__ == '__main__':
#     print(to_standard_amr(s1))
#     print(to_standard_amr(s2))

