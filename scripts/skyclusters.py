from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes
from experimentdata import ExperimentData, DataGetter
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt
from numpy import array, mean, percentile, std, zeros, full
from math import ceil
import json
import cPickle as pkl
import time
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF

class vs_dom_itr:
    def __init__(self, obj, itr, domid):
        self.obj = obj
        self.itr = itr
        self.domid = domid
    def __iter__(self):
        return self

    def next(self):
        a, b = next(self.itr)
        return (self.obj.nodes[a].results,
        self.obj.nodes[b].results,
        self.obj.matrix[self.obj.get_matrix_index(self.obj.nodes.posmap[a], self.obj.nodes.posmap[b])], self.domid)

def get_closeness((a, b, kwargs)):
    nc = NodeComparison(a, b, **kwargs)
    return nc.closeness

def get_vs_domain((a, b, distance, dom)):
    if dom in a and dom in b:
        match = float(len(a[dom].intersection(b[dom])))/float(len(a[dom].union(b[dom])))
        return True, dom, (1.0-distance) - match
    return (False, None, None)

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
    def prefix(self):
        return self.nodes.prefix

    @prefix.setter
    def prefix(self, val):
        self.nodes.prefix = val

    @prefix.setter
    def prefix(self, val):
        self._prefix = int(val)

    @property
    def chunksize(self):
        if not hasattr(self, '_chunksize'):
            self._chunksize = 20000
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

    def closeness_vs_domain(self, domid, workers=None):
        if not hasattr(self, '_closeness_vs_domain_stats'):
            if len(self.matrix) == 0:
                self.make_closeness_matrix(workers)
            print('comparing closeness to each domain')
            combos = itertools.combinations(range(len(self.nodes)), 2)
            itr = vs_dom_itr(self, combos, domid)
            D = DataGetter(prefix=self.prefix)
            domd = dict()
            if not self.nodes.limit:
                domd[D.int2dom(domid)] = full(len(self.matrix), -3.0, float)
            else:
                domd[D.int2dom(domid)] = full(1+((self.nodes.limit**2)/2), -3.0, float)
            dompos = defaultdict(int)
            count = 0
            pool = Pool(processes=workers)
            count = 0
            t = time.time()
            results = list()
            '''
            for item in itr:
                res = get_vs_domain(item)
            '''
            for res in pool.imap_unordered(get_vs_domain, itr, self.chunksize):
                if res[0]:
                    domd[D.int2dom(res[1])].put(dompos[res[1]], res[2])
                    dompos[res[1]] += 1
                if count % 10000 == 0:
                    print(str(count)+' --- '+str(time.time()-t))
                count += 1
            for dom in domd:
                domd[dom] = [z for z in domd[dom] if z > -3]
            with open(self.fmt_path('datadir/closeness_vs_dom/'+self.timeid+'/'+str(domid)+'.json'), 'w') as f:
                json.dump(domd, f)
            outd = dict()
            maxstd = 0
            for d in domd:
                tmp = [abs(z) for z in domd[d]]
                outd[d] = {
                        'mean': mean(tmp),
                        'median': percentile(tmp, 50),
                        '25': percentile(tmp, 25),
                        '75': percentile(tmp, 75),
                        'std': std(domd[d])
                        }
                if outd[d]['std'] > maxstd:
                    maxstd = outd[d]['std']
            with open(self.fmt_path('datadir/closeness_vs_dom/'+self.timeid+'/'+str(domid)+'.json'), 'w') as f:
                json.dump(outd, f)

            self._closeness_vs_domain_stats = outd
            self._closeness_vs_domain_raw = dict(domd)
            print(time.time() - t)

    def plot_closeness_vs_domain(self, *allstats):
        stats = dict()
        for dom in allstats:
            stats.update(dom)
        D = DataGetter(prefix=self.prefix)
        mean_errs = list()
        answers = list()
        stds = list()
        tups = list()
        for dom in stats:
            answers.append(D.diversity(dom))
            mean_errs.append(stats[dom]['mean'])
            stds.append(stats[dom]['std'])
            tups.append((mean_errs[-1], stds[-1], answers[-1]))

        tups = sorted(tups, key=lambda z: z[0])
        with open(self.fmt_path('datadir/closeness_vs_dom/'+self.timeid+'/err_vs_diversity.json'), 'w') as f:
            json.dump(tups, f)

        stds = [50.0*float(z)/float(maxstd) for z in stds]
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(mean_errs, answers, stds, alpha=0.3)
        ax.set_xlabel('mean closeness error')
        ax.set_ylabel('# distinct answers')
        fig.savefig(self.fmt_path('plotsdir/closeness_vs_dom/'+self.timeid+'.png'))
        plt.close(fig)
        print(self.fmt_path('plotsdir/closeness_vs_dom/'+self.timeid+'.png'))
        print('backing up objects...')
        self.save_self()
        return stats

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

    def get_pair_indices(self, k):
        '''
        k -> position in condensed distance matrix
        https://stackoverflow.com/a/36867493/4335446
        '''
        n = len(self.nodes)
        i = int(ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))
        tmp = i * (n - 1 - i) + (i*(i + 1))/2
        j = int(n - tmp + k)
        return (i, j)

    def get_matrix_index(self, i, j):
        '''
        i and j -> indices in nodes
        '''
        assert i != j, "no diagonal elements in condensed matrix"
        if i < j:
            i, j = j, i
        n = len(self.nodes)
        return n*j - j*(j+1)/2 + i - 1 - j

    def get_pair_closeness(self, i, j):
        return self.matrix[self.get_matrix_index(i, j)]

    def crne(self, i, j):
        return self.matrix[self.get_matrix_index(i, j)]


if __name__ == "__main__":

    b = SkyClusterBuilder(limit=500)
    with open(b.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        b.kwargs['counts'] = pkl.load(f)
    #print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)[0])
    with open(b.fmt_path('datadir/matrix/matrix.json'), 'r') as f:
        b.matrix = json.load(f)
    b.closeness_vs_domain(5)
