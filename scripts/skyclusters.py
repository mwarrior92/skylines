from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes
from experimentdata import ExperimentData
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt
from numpy import array
import json
import cPickle as pkl
import time

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
            self.nodes = Nodes(limit=limit, min_tests=min_tests, **kwargs)

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
            self._chunksize = 50
        return self._chunksize

    @chunksize.setter
    def chunksize(self, val):
        self._chunksize = val

    def make_closeness_matrix(self, workers=None):
        print('calculating closeness matrix')
        self.matrix = list()
        a = time.time()
        itr = itertools.combinations(range(len(self.nodes)), 2)
        itr = itertools.imap(lambda z: (self.nodes[z[0]], self.nodes[z[1]],
            self.kwargs), itr)

        pool = Pool(processes=workers)
        count = 0
        print(self.chunksize)
        for c in pool.imap(get_closeness, itr, self.chunksize):
            self.matrix.append(c)
            if not count % 1000:
                print('comparison: '+str(count)+', result: '+str(self.matrix[-1]))
            count += 1
        self.duration = time.time()-a
        print('duraiton: '+str(self.duration)+', chunk: '+str(self.chunksize))
        print('done calculating matrix')
        self.linkage = linkage(self.matrix, method='complete')
        self.cophenet = cophenet(self.linkage, self.matrix)[0]
        self.save_self()
        path = self.fmt_path('objectdir/linkage/'+self.timeid+'.json')
        with open(path, 'w') as f:
            json.dump(self.linkage.tolist(), f)
        path = self.fmt_path('objectdir/matrix/'+self.timeid+'.json')
        with open(path, 'w') as f:
            json.dump(self.matrix, f)

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
                labels=[self.nodes[i].country for i in range(len(self.nodes))],
                **kwargs)
        if fname is not None:
            self.dendrogram_fname = fname
        fig.savefig(self.dendrogram_fname)
        plt.close(fig)
        self.save_self()
        return self.dendrogram_fname, self.dendrogram

if __name__ == "__main__":

    b = SkyClusterBuilder(limit=40)
    with open(b.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        b.kwargs['counts'] = pkl.load(f)
    print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)[0])
