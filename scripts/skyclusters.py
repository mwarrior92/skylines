from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes, CollapsedNode
from experimentdata import ExperimentData
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt
from numpy import array
import json

def get_closeness((a, b, kwargs)):
    nc = NodeComparison(a, b, **kwargs)
    nc.get_closeness()
    return nc.closeness

class SkyClusterBuilder(ExperimentData):
    def __init__(self, limit=0, min_tests=160, **kwargs):
        self.matrix = list()
        self.kwargs = kwargs
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if 'from_file' in kwargs and kwargs['from_file']:
            path = self.fmt_path('objectdir/nodes/'+self.timeid+'.json')
            with open(path, 'r') as f:
                p = json.load(f)
            self.nodes = Nodes(**p)
            try:
                path = self.fmt_path('objectdir/linkage/'+timeid+'.json')
                with open(path, 'r') as f:
                    self.linkage = array(json.load(f))
            except:
                pass
        if not hasattr(self, 'nodes'):
            self.nodes = Nodes(limit=limit, min_tests=min_tests)

    def save_self(self, timeid=None):
        if timeid is None:
            timeid = self.timeid
        if hasattr(self, 'linkage'):
            path = self.fmt_path('objectdir/linkage/'+timeid+'.json')
            linkage = self.linkage.tolist()
            with open(path, 'w') as f:
                json.dump(linkage, f)
        super(type(self),self).save_self(timeid)

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
        self.cophenet = cophenet(self.linkage, self.matrix)[0]
        self.save_self()
        path = self.fmt_path('objectdir/linkage_and_matrix/'+self.timeid+'.json')
        with open(path, 'w') as f:
            json.dump({'matrix': self.matrix, 'linkage': self.linkage.tolist(),
                'cophenet': self.cophenet}, f)

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
        return self.dendrogram_fname, self.dendrogram

if __name__ == "__main__":
    b = SkyClusterBuilder(limit=40)
    print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)[0])
