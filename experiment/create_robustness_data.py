import logging
import random
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List, Any

import networkx
import penman
from penman import load as penman_load, Graph, decode, encode
from penman.graph import CONCEPT_ROLE
from penman.models.amr import model as penman_model
from penman.transform import reify_edges, dereify_edges
from tqdm import tqdm


def _key_for_graph_conections(graph: Graph) -> str:
    nodes = {i.source: i.target for i in graph.instances()}
    triples = []
    for t in graph.edges():
        triples.append((nodes.get(t.source, t.source), t.role, nodes.get(t.target, t.target)))
    for t in graph.attributes():
        triples.append((nodes.get(t.source, t.source), t.role, t.target))

    triples = sorted(triples)
    return '\n'.join(' '.join(t) for t in triples)


def create_graph(triples, top=None, meta_id=None):
    if triples:
        return encode(Graph(triples, top=top, metadata={'id': meta_id}), model=penman_model, indent=4)
    else:
        return None


def create_connected_graph(triples, top=None, meta_id=None):
    try:
        # Connect as a multi-sentence if disconnected
        net = networkx.DiGraph()
        net.add_edges_from([(s, t) for s, _, t in triples])
        components = list(networkx.weakly_connected_components(net))
        if len(components) > 1:
            top_node = 't000'
            triples.append((top_node, CONCEPT_ROLE, 'multi-sentence'))

            # Add an edge as a multi-sentence result
            for i, nodeset in enumerate(components):
                component_top = random.choice([node for node in nodeset if net.out_degree(node)])
                triples.append((top_node, f':snt{i + 1}', component_top))

        # Remove duplicates
        triples = sorted(set(triples))
        return create_graph(triples, top, meta_id=meta_id)
    except:
        return None


def remove_duplicate(graph: Graph) -> Graph:
    new_triples = sorted(set(graph.triples))
    return decode(create_graph(new_triples, top=graph.top, meta_id=graph.metadata['id']), model=penman_model)


def rearrange_random(graph: Graph):
    tree = penman.layout.reconfigure(graph, model=penman_model)
    penman.layout.rearrange(tree, key=penman_model.random_order)
    # tree.reset_variables('{prefix}{j}')
    return penman._format.format(tree, indent=4)


def _duplicate(t: Tuple[str, List[Tuple[str, Any]]], n_dup: int = 0):
    edges = []
    for role, tgt in t[1]:
        if not penman._format.is_atomic(tgt):
            tgt, n_dup = _duplicate(tgt, n_dup)

        edges.append((role, tgt))
        if role != '/':
            tgt_var = tgt if penman._format.is_atomic(tgt) else tgt[0]
            edges.append((role, tgt_var))
            n_dup += 1

    return (t[0], edges), n_dup


def duplicate_edge(graph: Graph):
    tree = penman.layout.configure(graph, model=penman_model)
    edges, n_dup = _duplicate(tree.node)
    new_tree = penman.Tree(edges, metadata=tree.metadata)
    if n_dup > 0:
        return penman._format.format(new_tree, indent=2)
    else:
        return None


def relabel_node(graph: Graph):
    tree = penman.layout.reconfigure(graph, model=penman_model)
    letter = random.choice(string.ascii_lowercase)
    tree.reset_variables(fmt=letter+'{i}')
    return penman._format.format(tree, indent=4)


def change_top(graph: Graph):
    variables = [v for v in graph.variables() if v != graph.top]
    if variables:
        new_top = random.choice(variables)
        tree = penman.layout.reconfigure(graph, top=new_top, model=penman_model)
        return penman._format.format(tree, indent=4)
    else:
        return None


def insert_one_edge(graph: Graph):
    instances = graph.instances()
    source = random.choice(instances).source
    target = random.choice(instances).source

    return create_graph(graph.triples + [(source, ':prep-test', target)], top=graph.top, meta_id=graph.metadata['id'])


def insert_one_inst(graph: Graph):
    if random.random() > 0.5:
        instances = graph.instances()
        source = random.choice(instances).source

        return create_graph(graph.triples + [('x327', CONCEPT_ROLE, 'X327'), (source, ':prep-test', 'x327')],
                            top=graph.top, meta_id=graph.metadata['id'])
    else:
        instances = graph.instances()
        source = random.choice(instances).source

        return create_graph(graph.triples + [(source, ':quant', '0.318889172')], top=graph.top,
                            meta_id=graph.metadata['id'])


def change_one_edge(graph: Graph):
    try:
        triples = graph.edges()
        pick = random.choice(triples)
        changed = (pick.source, ':prep-test', pick.target)

        return create_graph([t if t != pick else changed
                             for t in graph.triples], top=graph.top, meta_id=graph.metadata['id'])
    except:
        return None


def change_one_inst(graph: Graph):
    instances = [t for t in graph.instances() + graph.attributes() if t.role != ':wiki']
    pick = random.choice(instances)
    changed = (pick.source, pick.role, 'random-ghost' if pick.role == CONCEPT_ROLE else '"Random Ghost"')

    return create_graph([t if t != pick else changed
                         for t in graph.triples], top=graph.top, meta_id=graph.metadata['id'])


def swap_two_edge(graph: Graph):
    try:
        edges = list(graph.edges())
        a = random.choice(edges)

        edges = [e for e in edges if e.source != a.source and e.role != a.role and e.target != a.target]
        b = random.choice(edges)

        replace_dict = {a: (a.source, b.role, b.target),
                        b: (b.source, a.role, a.target)}

        return create_connected_graph([replace_dict.get(t, t)
                                       for t in graph.triples], top=graph.top, meta_id=graph.metadata['id'])
    except:
        return None


def delete_one_edge(graph: Graph):
    try:
        triples = list(graph.edges())
        pick = random.choice(triples)

        return create_connected_graph([t for t in graph.triples
                                       if t != tuple(pick)], top=graph.top, meta_id=graph.metadata['id'])
    except:
        return None


def delete_one_inst(graph: Graph):
    if random.random() > 0.5:
        try:
            triples = [t for t in graph.attributes() if t.role != ':wiki']
            pick = random.choice(triples)

            return create_connected_graph([t for t in graph.triples
                                           if t != tuple(pick)], top=graph.top, meta_id=graph.metadata['id'])
        except:
            return None
    else:
        instances = [t for t in graph.instances() if t.target != 'multi-sentence']
        pick = random.choice(instances).source

        return create_connected_graph([t for t in graph.triples
                                       if not (t[0] == pick or (t[2] == pick and t[1] != CONCEPT_ROLE))],
                                      top=graph.top if graph.top != pick[0] else None, meta_id=graph.metadata['id'])


if __name__ == '__main__':
    import penman.transform
    import penman.layout

    penman.transform.logger.setLevel(logging.WARN)
    penman.layout.logger.setLevel(logging.WARN)

    parser = ArgumentParser()
    parser.add_argument('--reference', '-ref', '-r', type=Path, nargs='+',
                        help='List of reference AMR annotation files')
    parser.add_argument('--output', '-out', '-o', type=Path, help='Directory to save variation files')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Seed for generating variations')
    args = parser.parse_args()
    random.seed(args.seed)

    if not args.output.exists():
        args.output.mkdir(parents=True)

    graphs = []
    graph_specs = set()
    for file in args.reference:
        for amr in tqdm(penman_load(file, model=penman_model), desc=f'{file}'):
            amr = remove_duplicate(amr)

            spec = _key_for_graph_conections(amr)
            if spec in graph_specs:
                # Don't add a graph with previously seen connection specs
                continue

            graph_specs.add(spec)
            graphs.append({
                'original': encode(amr, model=penman_model),

                'eq_new_root': change_top(amr),
                'eq_reorder': rearrange_random(amr),
                'eq_relabel': relabel_node(amr),
                'eq_reify': encode(reify_edges(amr, model=penman_model), model=penman_model),
                'eq_dereify': encode(dereify_edges(amr, model=penman_model), model=penman_model),
                'eq_duplicate': duplicate_edge(amr),

                'ne_insert_node': insert_one_inst(amr),
                'ne_insert_edge': insert_one_edge(amr),
                'ne_change_node': change_one_inst(amr),
                'ne_change_edge': change_one_edge(amr),
                'ne_delete_node': delete_one_inst(amr),
                'ne_delete_edge': delete_one_edge(amr),
                'ne_swap': swap_two_edge(amr)
            })

    # Filter-out not modified things
    graphs = [gdict for gdict in graphs if all(v is not None for v in gdict.values())]

    # Sample 20,000 data (due to power test result)
    # At sig.level 0.0001, the difference between 0.0 and 0.001 with power 0.9999 requires 13,826 samples
    random.shuffle(graphs)
    graphs = graphs[:20_000]
    print(f'Selected {len(graphs)} graphs.')

    for key in graphs[0].keys():
        with (args.output / f'{key}.txt').open('w+t') as fp:
            fp.write('\n\n'.join([graph[key] for graph in graphs]))