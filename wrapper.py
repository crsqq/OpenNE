from OpenNE.src.libnrl import graph
from OpenNE.src.libnrl import grarep
from OpenNE.src.libnrl import line
from OpenNE.src.libnrl import node2vec
from OpenNE.src.libnrl.gcn import gcnAPI
from itertools import product
import networkx as nx
import numpy as np


def nx_to_openne_graph(nxgraph, stringify_nodes=True):
    dg = nx.to_directed(nxgraph).copy()
    if stringify_nodes:
        nx.relabel_nodes(dg, {n:str(n) for n in dg.nodes}, copy=False)
    nx.set_edge_attributes(dg, 1.0, 'weight')
    g = graph.Graph()
    g.G = dg
    g.encode_node()
    return g


class OpenNEEmbeddingBase:
    def __init__(self, thisgraph, parameters):
        self.graph = nx_to_openne_graph(thisgraph)
        self.embeddings = None
        self.parameters = parameters
    def run(self):
        raise NotImplementedError('')
    def update_parameters(self, new_parameters):
        self.parameters = new_parameters
        self.embeddings = None
    def get_embeddings(self):
        if not self.embeddings:
            self.run()
        return self.embeddings
    def get_vectors(self):
        return self.get_embeddings().vectors

class Node2VecEmbedding(OpenNEEmbeddingBase):
    def run(self):
        self.embeddings = node2vec.Node2vec(self.graph, **self.parameters)

class GraRepEmbedding(OpenNEEmbeddingBase):
    def run(self):
        self.embeddings = grarep.GraRep(self.graph, **self.parameters)

class LINEEmbedding(OpenNEEmbeddingBase):
    def run(self):
        self.embeddings = line.LINE(self.graph, **self.parameters)
