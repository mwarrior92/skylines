from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes, CollapsedNode
from experimentdata import ExperimentData
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def get_closeness((pos, nodes, kwargs)):
    i, j = pos
    nc = NodeComparison(i, j, nodes, **kwargs)
    nc.get_closeness()
    print(str(pos)+': '+str(nc.closeness))
    return nc.closeness

class SkyClusterBuilder(ExperimentData):
    def __init__(self, workers=None, limit=None, min_tests=160, **kwargs):
        self.nodes = Nodes(limit=limit, min_tests=min_tests)
        self.threads = list()
        self.matrix = list()
        self.pool = Pool(processes=workers)
        self.kwargs = kwargs
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def chunksize(self):
        if not hasattr(self, '_chunksize'):
            self._chunksize = min([len(self.nodes)/4, 100])
            if not self._chunksize:
                self._chunksize = 1
        return self._chunksize

    @chunksize.setter
    def chunksize(self, val):
        self._chunksize = val

    def make_closeness_matrix(self):
        print('calculating closeness matrix')

        itr = itertools.combinations(range(len(self.nodes)), 2)
        itr = itertools.izip(itr, itertools.repeat(self.nodes),
                itertools.repeat(self.kwargs))

        for c in self.pool.imap(get_closeness, itr, self.chunksize):
                self.matrix.append(c)
        print('done calculating matrix')

    @property
    def dendrogram_fname(self):
        path = self.fmt_path('plotsdir/dendrogram/'+self.timestr+'.png')
        return path

    def make_dendrogram(self, fname=None, **kwargs):
        if len(self.matrix) == 0:
            self.make_closeness_matrix()
        print('making dendrogram')
        L = linkage(self.matrix, method='complete')
        fig, (ax) = plt.subplots(1,1, figsize=(6, 3.5))
        self.dendrogram = dendrogram(L, ax=ax,
                labels=[CollapsedNode(self.nodes[i]).country for i in range(len(self.nodes))],
                **kwargs)
        if fname is None:
            fname = self.dendrogram_fname
        fig.savefig(fname)
        plt.close(fig)
        return fname

if __name__ == "__main__":
    b = SkyClusterBuilder(limit=30)
    print(b.make_dendrogram())
