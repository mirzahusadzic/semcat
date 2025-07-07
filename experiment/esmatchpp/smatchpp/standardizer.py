from typing import Tuple, List
from collections import Counter, defaultdict

import penman
from penman.codec import format
from penman.layout import configure, rearrange, interpret, reconfigure
from penman.models.amr import model, reifications
from penman.transform import dereify_edges

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

# r1 = '(a / and :op1 (f / find-01:ARG1 (p / photo:ARG0-of (h2 / have-03:ARG1 (t / thing:ARG2-of (n2 / name-01:ARG1 p2):ARG1-of (w / write-01:location b))):part (b / back):topic (p2 / person:ARG0-of (h / have-rel-role-91:ARG1 (p3 / person:name (n / name:op1 "Frizell"):wiki -):ARG2 (g / granddaughter)))):location (r / relative-position:op1 (h3 / home:poss p3):quant (s / several:op1 (d / distance-quantity:quant 1:unit (m / mile))))):op2 (g2 / give-01:ARG1 p:ARG2 (s2 / station:ARG1-of (l / local-02):mod (t2 / television))))'
# r2 = '(a / and :op1 (f / find-01:ARG1 (p / photo:poss (t / thing:ARG2-of (n2 / name-01:ARG1 p2):ARG1-of (w / write-01:location b)):part (b / back):topic (p2 / person:ARG0-of (h / have-rel-role-91:ARG1 (p3 / person:name (n / name:op1 "Frizell"):wiki -):ARG2 (g / granddaughter)))):location (r / relative-position:op1 (h3 / home:poss p3):quant (s / several:op1 (d / distance-quantity:quant 1:unit (m / mile))))):op2 (g2 / give-01:ARG1 p:ARG2 (s2 / station:ARG1-of (l / local-02):mod (t2 / television))))'
#
# if __name__ == '__main__':
#     print(to_standard_amr(r1))
#     print(to_standard_amr(r2))