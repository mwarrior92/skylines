from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes, CollapsedNode
from experimentdata import ExperimentData
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt

def get_closeness((a, b, kwargs)):
    nc = NodeComparison(a, b, **kwargs)
    nc.get_closeness()
    return nc.closeness

class SkyClusterBuilder(ExperimentData):
    def __init__(self, limit=0, min_tests=160, **kwargs):
        self.threads = list()
        self.matrix = list()
        self.kwargs = kwargs
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if not hasattr(self, 'nodes'):
            self.nodes = Nodes(limit=limit, min_tests=min_tests)

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

    def make_closeness_matrix(self, workers=None):
        print('calculating closeness matrix')

        itr = itertools.combinations(range(len(self.nodes)), 2)
        itr = itertools.imap(lambda z: (CollapsedNode(self.nodes[z[0]]),
            CollapsedNode(self.nodes[z[1]]),
            self.kwargs), itr)

        pool = Pool(processes=workers)
        count = 0
        for c in pool.imap(get_closeness, itr, self.chunksize):
            self.matrix.append(c)
            count += 1
            if not count % 1000:
                print(self.matrix[-1])
        print('done calculating matrix')
        self.linkage = linkage(self.matrix, method='complete')
        self.cophenet = cophenet(self.linkage, self.matrix)
        self.save_self()

    @property
    def dendrogram_fname(self):
        if not hasattr(self, '_dendrogram_fname'):
            self._dendrogram_fname = self.fmt_path('plotsdir/dendrogram/'+self.timeid+'.png')
        return self._dendrogram_fname

    @dendrogram_fname.setter
    def dendrogram_fname(self, val):
        self._dendrogram_fname = val

    def make_dendrogram(self, fname=None, **kwargs):
        if not hasattr(self, 'linkage'):
            self.make_closeness_matrix()
        print('making dendrogram')
        fig, (ax) = plt.subplots(1,1, figsize=(6, 3.5))
        self.dendrogram = dendrogram(self.linkage, ax=ax,
                labels=[CollapsedNode(self.nodes[i]).country for i in range(len(self.nodes))],
                **kwargs)
        if fname is not None:
            self.dendrogram_fname = fname
        fig.savefig(self.dendrogram_fname)
        plt.close(fig)
        self.save_self()
        return self.dendrogram_fname

if __name__ == "__main__":
    b = SkyClusterBuilder(limit=40)
    #print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50))
    b.make_closeness_matrix()
