from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes
from experimentdata import ExperimentData, DataGetter
from multiprocessing import Pool
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt
from numpy import array, mean, percentile, std, zeros, full
import json
import cPickle as pkl
import time
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF

def get_closeness((a, b, kwargs)):
    nc = NodeComparison(a, b, **kwargs)
    return nc.closeness

def get_vs_domain((a, b, closeness, kwargs)):
    return nc.closeness_vs_domain

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

    def closeness_vs_domain(self, workers=None):
        if not hasattr(self, '_closeness_vs_domain_stats'):
            if len(self.matrix) == 0:
                self.make_closeness_matrix(workers)
            t = time.time()
            print('comparing closeness to each domain')
            itr = itertools.izip(itertools.combinations(range(len(self.nodes)), 2),
                    xrange(len(self.matrix)))
            itr = itertools.imap(lambda (z, x): (self.nodes[z[0]].results, self.nodes[z[1]].results, self.matrix[x]), itr)
            D = DataGetter(prefix=self.prefix)
            domd = dict()
            for dom in D.test_counts:
                if not self.nodes.limit:
                    domd[D.int2dom(dom)] = full(int(D.test_counts[dom])**2, -3.0, float)
                else:
                    domd[D.int2dom(dom)] = full(self.nodes.limit**2, -3.0, float)
            dompos = defaultdict(int)
            count = 0
            for a, b, c in itr:
                for d  in set(a.keys()).intersection(b.keys()):
                    dom = D.int2dom(d)
                    pos = dompos[d]
                    print(domd[dom][pos])
                    domd[dom].put(pos, (1-c) - float(len(a[d].intersection(b[d])))/len(a[d].union(b[d])))
                    print(dom)
                    print(pos)
                    print(domd[dom][pos])
                    dompos[d] += 1
                if count % 10000 == 0:
                    print(time.time())
                count += 1
            for dom in domd:
                domd[dom] = [z for z in domd[dom] if z > -3]
            with open(self.fmt_path('datadir/closeness_vs_dom/'+self.timeid+'/raw.json'), 'w') as f:
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
            with open(self.fmt_path('datadir/closeness_vs_dom/'+self.timeid+'/stats.json'), 'w') as f:
                json.dump(outd, f)

            self._closeness_vs_domain_stats = outd
            self._closeness_vs_domain_raw = dict(domd)
            print(time.time() - t)
        stats = self._closeness_vs_domain_stats
        d = DataGetter(prefix=self.prefix)
        mean_errs = list()
        answers = list()
        stds = list()
        tups = list()
        for dom in stats:
            answers.append(d.diversity(dom))
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
        self.save_self()
        print(self.fmt_path('plotsdir/closeness_vs_dom/'+self.timeid+'.png'))
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

if __name__ == "__main__":

    b = SkyClusterBuilder()
    with open(b.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        b.kwargs['counts'] = pkl.load(f)
    #print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)[0])
    with open(b.fmt_path('datadir/matrix/matrix.json'), 'r') as f:
        b.matrix = json.load(f)
    b.closeness_vs_domain()
