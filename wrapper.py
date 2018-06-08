from OpenNE.src.libnrl import graph
from OpenNE.src.libnrl import grarep
from OpenNE.src.libnrl import line
from OpenNE.src.libnrl import node2vec
from OpenNE.src.libnrl.gcn import gcnAPI
from itertools import product
import networkx as nx
import numpy as np
import tensorflow as tf

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

    @staticmethod
    def valid_parameter_combinations(parameterSpace):
        """
        returns all possible combinations, if some are not valid / useful,
        this method needs to be overwritten
        """
        all_combinations = product(*parameterSpace.values())
        return [{k:v for k,v in zip(parameterSpace.keys(), combn)} for combn in all_combinations]

class Node2VecEmbedding(OpenNEEmbeddingBase):
    """
     {'dim': 2, 'num_paths': 80, 'p': 1, 'path_length': 10, 'q': 1}
    """
    def run(self):
        self.embeddings = node2vec.Node2vec(self.graph, retrainable=True, **self.parameters)

    def retrain(self, new_graph):
        g = nx_to_openne_graph(new_graph)
        self.embeddings.retrain(g)

class GraRepEmbedding(OpenNEEmbeddingBase):
    def run(self):
        self.embeddings = grarep.GraRep(self.graph, **self.parameters)

    @staticmethod
    def valid_parameter_combinations(parameterSpace):
        """
        returns all possible combinations, if some are not valid / useful,
        this method needs to be overwritten
        """
        all_combinations = product(*parameterSpace.values())
        all_combinations = [{k:v for k,v in zip(parameterSpace.keys(), combn)} for combn in all_combinations]
        return [x for x in all_combinations if x["dim"] % x["Kstep"] == 0]

class LINEEmbedding(OpenNEEmbeddingBase):
    def run(self):
        tf.reset_default_graph()
        self.embeddings = line.LINE(self.graph, **self.parameters)



from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

class SpectralClusteringEmbedding(OpenNEEmbeddingBase):
    def __init__(self, thisgraph, parameters):
        self.graph = thisgraph
        self.embeddings = None
        self.parameters = parameters

        nx.relabel_nodes(self.graph, {n:str(n) for n in self.graph.nodes}, copy=False)
    def run(self):
        L = nx.normalized_laplacian_matrix(self.graph)
        evalues, evectors = a,b = largest_eigsh(L, k=self.parameters['dim'])
        self.embeddings = {str(n):v for n,v in zip(self.graph.nodes, evectors)}
    def get_vectors(self):
        return self.get_embeddings()


def _RandNE(graph, dim, q, beta):
    d = dim
    A = nx.to_scipy_sparse_matrix(graph)

    R = np.random.normal(loc=0, scale=1/d, size=(A.shape[0], d))

    U0, _  = np.linalg.qr(R)

    Ulist = [U0]
    for i in range(q):
        Ulist.append(A.dot(Ulist[-1]))

    Ulist = np.array(Ulist)

    betas = (beta**np.arange(0, q+1))

    U = np.array([scalar*m for scalar,m in zip(betas, Ulist)]).sum(axis=0)
    return U

class RandNEEmbedding(OpenNEEmbeddingBase):
    def __init__(self, thisgraph, parameters):
        self.graph = thisgraph
        self.embeddings = None
        self.parameters = parameters
    def run(self):
        U = _RandNE(self.graph, **self.parameters)
        self.embeddings = {str(n):v for n,v in zip(self.graph.nodes, U)}
    def get_vectors(self):
        return self.get_embeddings()
