from __future__ import print_function
import time
from gensim.models import Word2Vec
from . import walker


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, retrainable=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        self.dw = dw
        self.p =p
        self.q = q
        self.kwargs = kwargs
        self.retrainable = retrainable

        self.path_length = path_length
        self.num_paths = num_paths

        self._init_walker()

        sentences = self.walker.simulate_walks(num_walks=self.num_paths, walk_length=self.path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        #print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = self._extract_vectors(word2vec)

        if not retrainable:
            del word2vec
        else:
            self.word2vec = word2vec

    def retrain(self, new_graph, num_paths=80, epochs=5):
        if not self.retrainable:
            raise RuntimeError('model is not retrainable')

        self.graph = new_graph
        self._init_walker()

        sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=self.path_length)
        kwargs = self.kwargs
        kwargs["sentences"] = sentences

        self.word2vec.train(sentences=sentences, total_examples=self.word2vec.corpus_count, epochs=epochs)
        self.vectors = self._extract_vectors(self.word2vec)


    def _extract_vectors(self, w2v):
        return {n:w2v.wv[n] for n in self.graph.G.nodes()}

    def _init_walker(self):
        if self.dw:
            self.walker = walker.BasicWalker(self.graph, workers=self.kwargs["workers"])
        else:
            self.walker = walker.Walker(self.graph, p=self.p, q=self.q, workers=self.kwargs["workers"])
            #print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                ' '.join([str(x) for x in vec])))
            fout.close()

