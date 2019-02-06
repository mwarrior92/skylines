import os
from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes
from experimentdata import ExperimentData
from multiprocessing import Pool
import itertools
import cPickle as pkl

class closenessMatrixMaker(ExperimentData):
    def __init__(self, workers=None, limit=None, **kwargs):
        self.nodes = Nodes(limit=limit)
        self.threads = list()
        self.matrix = list()
        self.pool = Pool(processes=workers)
        self.kwargs = kwargs
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def counts(self):
        if not hasattr(self, '_counts') or self._counts is None:
            path = self.fmt_path('datadir/pkls/answer_counts.pkl')
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    self._counts = pkl.load(f)
            else:
                self._counts = count_answers_across_nodes(self.nodes)
                with open(path, 'w') as f:
                    pkl.dump(self._counts, f)
        return self._counts

    @counts.setter
    def counts(self, val):
        self._counts = val

    def make_closeness_matrix(self):
        print('calculating closeness matrix')
        def get_closeness(pos):
            i, j = pos
            nc = NodeComparison(i, j, self.nodes, self.counts, **self.kwargs)
            nc.get_closeness()
            return nc

        for c in self.pool.imap(get_closeness,
                itertools.combinations(range(len(self.nodes))), 10):
            self.matrix.append(c)

    def get_closeness_matrix(self):
        if len(self.matrix) == 0:
            self.make_closeness_matrix()
        return self.matrix
