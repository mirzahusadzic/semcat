import networkx
import networkx as nx
import pandas as pd
import penman
from penman.codec import format
from penman.layout import interpret, reconfigure
from penman.models.amr import model
from penman.transform import dereify_edges

count = 0
reify = []
depths = []

def get_max_depth(graph):
    max_depth = 0
    tree = reconfigure(graph, top=graph.top, model=model)
    for branch in tree.walk():
        if branch[0][0] > max_depth:
            max_depth = branch[0][0]
    return max_depth

def dereify(graph) -> str:
    global count
    global reify
    tree = reconfigure(graph, top=graph.top, model=model)
    dereified_graph = dereify_edges(interpret(tree, model=model), model=model)
    if graph != dereified_graph:
        count+=1
        reify.append(data.index(graph))
        # print(f"this graph has applied reification: {data.index(graph)}")
    # Convert it into a tree and rearrange
    tree = reconfigure(dereified_graph, top=graph.top, model=model)
    return format(tree).strip()

data = penman.load('/mnt/sda/Code_local/RoSE/experiment/resources/amr-annotations/amr3.0-test.txt', model=model, encoding='utf-8')
for item in data:
    dereify(item)
    depths.append(get_max_depth(item))

avg_depth = sum(depths)/len(depths)
d_count = 0
for depth in depths:
    if depth < 6:
        d_count+=1

print(f"Coverage of depth 5: {d_count/len(depths)}")
print(f"Average depth: {avg_depth}")
print(f"Total reify-applied sentences: {count}")

ids = []

for idx in reify:
    ids.append(data[idx].metadata['id'])

result = pd.read_csv('result/per-item.csv')
v3_data = result[result['name'].str.startswith('v3.0_')]
v3_data = v3_data[v3_data['item'].isin(ids)]

smatch = list(v3_data[(v3_data['name']=='v3.0_AMRBART-v2-3.0')&(v3_data['metric']=='SMATCH')]['score'])
ssharp = list(v3_data[(v3_data['name']=='v3.0_AMRBART-v2-3.0')&(v3_data['metric']=='ESMATCH++mac')]['score'])

d_count = 0
for i in range(0, len(smatch)):
    if smatch[i] != ssharp[i]:
        d_count+=1

print(d_count)

print(v3_data[(v3_data['name']=='v3.0_AMRBART-v2-3.0')&(v3_data['metric']=='SMATCH')])
print(v3_data[(v3_data['name']=='v3.0_AMRBART-v2-3.0')&(v3_data['metric']=='ESMATCH++mac')])
