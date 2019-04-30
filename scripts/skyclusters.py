from skycompare import NodeComparison, count_answers_across_nodes
from skynodes import Nodes
from experimentdata import ExperimentData, DataGetter
from multiprocessing import Pool, Process, Queue
from multiprocessing.sharedctypes import RawArray
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
import json
import cPickle as pkl
import time
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF
from ctypes import c_double
import os.path
import gc


def cycle_worker(q):
    while True:
        job, arg = q.get()
        if type(job) is str and job == 'exit':
            return True
        job(arg)
        del arg

def dump_domain_error(data):
    out = defaultdict(list)
    for dom, val in data:
        out[dom].append(val)
    D = DataGetter()
    if not os.path.exists(D.datadir+'domain_error/'):
        os.makedirs(D.datadir+'domain_error/')

    for dom in out:
        with open(D.datadir+'domain_error/'+str(dom)+'.json', 'a') as f:
            f.write(json.dumps(out[dom])+'\n')

def get_pair_indices(k, n):
    '''
    k -> position in condensed distance matrix
    https://stackoverflow.com/a/36867493/4335446
    '''
    i = int(ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))
    I = i + 1
    tmp = I * (n - 1 - I) + (I*(I + 1))/2
    j = int(n - tmp + k)
    return (i, j)

def get_matrix_index(i, j, n):
    '''
    i and j -> indices in nodes
    '''
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n*j - j*(j+1)/2 + i - 1 - j

class domain_error_itr:
    def __init__(self, obj, itr):
        self.obj = obj
        self.itr = itr
        self.nodes = obj.nodes
        gc.collect()
    def __iter__(self):
        return self

    def next(self):
        a, b = next(self.itr)
        return (self.nodes[a].results,
            self.nodes[b].results,
            g_matrix[get_matrix_index(
                self.nodes.posmap[a],
                self.nodes.posmap[b],
                len(self.nodes))]
            )

def get_closeness((a, b, kwargs)):
    nc = NodeComparison(a, b, **kwargs)
    return nc.closeness

def get_domain_error((a, b, distance)):
    data = list()
    for dom in set(a.keys()).intersection(b.keys()):
        match = float(len(a[dom].intersection(b[dom])))/float(len(a[dom].union(b[dom])))
        data.append((dom, match - (1.0-distance)))
    return data

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
                    self.linkage = np.array(json.load(f))
            except:
                pass
        if not hasattr(self, 'nodes'):
            self.nodes = Nodes(limit=limit, min_tests=min_tests, **kwargs)

    def load_matrix_from_file(self, fname):
        with open(self.fmt_path(fname), 'r') as f:
            tmpmatrix = json.load(f)
            global g_matrix
            g_matrix = RawArray('d', len(tmpmatrix))
            for i, z in enumerate(tmpmatrix):
                g_matrix[i] = z
            self.matrix = g_matrix
            del tmpmatrix

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
            self._chunksize = 1000
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

    def domain_error(self, workers=3):
        print('comparing closeness to each domain')
        combos = itertools.combinations(range(len(self.nodes)), 2)
        itr = domain_error_itr(self, combos)
        data = np.full(len(self.matrix), -5.0)
        i = 0
        pool0 = Pool(workers-1)
        t = time.time()
        data = list()
        q = Queue()
        dumper = Process(target=cycle_worker, args=(q,))
        dumper.start()
        total = 0
        count = 0
        for res in pool0.imap_unordered(get_domain_error, itr, self.chunksize):
            data += res
            count += 1
            if count == 10000:
                total += len(data)
                print('total: '+str(total))
                q.put((dump_domain_error, data))
                count = 0
                data = list()
                gc.collect()
        if len(data):
            total += len(data)
            q.put((dump_domain_error, data))
        print('waiting to finish dumping')
        q.put(('exit', True))
        print('joining worker thread')
        dumper.join()
        print('joined worker thread; done')
        t2 = time.time()
        print(t2 - t)

    def condense_domain_error(self):
        D = DataGetter(prefix=self.prefix)
        data = dict()
        for dom in D.test_counts.keys():
            uncondensed = list()
            with open(D.datadir+'domain_error/'+str(dom)+'.json', 'r') as f:
                for line in f:
                    uncondensed += json.loads(line)
            if len(uncondensed):
                condensed = {
                        'std': np.std(uncondensed),
                        'raw_mean': np.mean(uncondensed),
                        'raw_median': np.median(uncondensed),
                        'diversity': D.diversity(dom)
                        }
                uncondensed = [abs(z) for z in uncondensed]
                condensed['abs_mean'] = np.mean(uncondensed)
                condensed['abs_median'] = np.median(uncondensed)
                condensed['25'] = np.percentile(uncondensed, 25)
                condensed['75'] = np.percentile(uncondensed, 75)
                data[dom] = condensed
        with open(D.fmt_path('datadir/domain_error/summary.json'), 'w') as f:
            json.dump(data, f)

    def plot_domain_error(self, *allstats):
        D = DataGetter()
        with open(D.fmt_path('datadir/domain_error/summary.json'), 'r') as f:
            stats = json.load(f)
        mean_errs = list()
        answers = list()
        stds = list()
        tups = list()
        for dom in stats:
            mean_errs.append(stats[dom]['abs_mean'])
            stds.append(stats[dom]['std'])
            answers.append(stats[dom]['diversity'])

        stds = [50.0*float(z)/float(max(stds)) for z in stds]
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(mean_errs, answers, stds, alpha=0.3)
        ax.set_xlabel('mean closeness error')
        ax.set_ylabel('# distinct answers')
        fig.savefig(self.fmt_path('plotsdir/domain_error/'+self.timeid+'.png'))
        plt.close(fig)
        print(self.fmt_path('plotsdir/domain_error/'+self.timeid+'.png'))

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

    def get_pair_indices(self, k, n=None):
        '''
        k -> position in condensed distance matrix
        https://stackoverflow.com/a/36867493/4335446
        '''
        if not n:
            n = len(self.nodes)
        i = int(ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))
        tmp = i * (n - 1 - i) + (i*(i + 1))/2
        j = int(n - tmp + k)
        return (i, j)

    def get_matrix_index(self, i, j, n=None):
        '''
        i and j -> indices in nodes
        '''
        assert i != j, "no diagonal elements in condensed matrix"
        if i < j:
            i, j = j, i
        if not n:
            n = len(self.nodes)
        return n*j - j*(j+1)/2 + i - 1 - j

    def get_pair_closeness(self, i, j):
        return self.matrix[self.get_matrix_index(i, j)]

    def cnre(self, i, j):
        A = self.nodes.posmap[i]
        B = self.nodes.posmap[j]
        return self.matrix[self.get_matrix_index(A, B)]

    def reduce_matrix_to_sampled(self):
        ''' TODO: IMPORTANT hard coded in the number of nodes; take this out if reusing code on new dataset '''
        print('reducing matrix')
        if len(self.nodes) == 9024:
            return
        matrix = np.zeros(len(list(itertools.combinations(range(len(self.nodes)), 2))))
        for a,b in itertools.combinations(range(len(self.nodes)), 2):
            i = get_matrix_index(a,b,len(self.nodes))
            A = self.nodes.posmap[a]
            B = self.nodes.posmap[b]
            I = get_matrix_index(A,B,9024)
            matrix[i] = self.matrix[i]
        del self.matrix
        self.matrix = matrix
        for i in range(len(self.nodes)):
            self.nodes.posmap[i] = i



if __name__ == "__main__":

    g_scb = SkyClusterBuilder(limit=500)

    ''' get matrix and make dendrogram
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    #print(b.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)[0])
    '''

    ''' get domain error '''
    with open(g_scb.fmt_path('datadir/matrix/matrix.json'), 'r') as f:
        tmpmatrix = json.load(f)
        g_matrix = RawArray('d', len(tmpmatrix))
        for i, z in enumerate(tmpmatrix):
            g_matrix[i] = z
        g_scb.matrix = g_matrix
        del tmpmatrix
    gc.collect()
    g_scb.domain_error()
    g_scb.condense_domain_error()
    g_scb.plot_domain_error()
