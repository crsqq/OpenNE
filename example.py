from src.libnrl import node2vec, graph
import networkx as nx


def nx_to_openne_graph(nxgraph):
    dg = nx.to_directed(nxgraph)
    nx.set_edge_attributes(dg, 1.0, 'weight')
    g = graph.Graph()
    g.G = dg
    g.encode_node()
    return g


# 2d embeddings for zachary network
zachary = nx.karate_club_graph()
zachary = nx.relabel_nodes(zachary, {n:str(n) for n in zachary.nodes})

embeddings = node2vec.Node2vec(nx_to_openne_graph(zachary), 10, 80, 2).vectors
print(embeddings)
