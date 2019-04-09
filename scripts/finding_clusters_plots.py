import skyclusters
from skycompare import NodeComparison
import json
import cPickle as pkl
from multiprocessing import Pool, Process, RawArray, Queue
from skypings import Pings
import numpy as np
import itertools
from experimentdata import DataGetter
import clusterAnalysis
import skyresolvers

def make_dendrogram():
    scb = skyclusters.SkyClusterBuilder(limit=500)

    with open(scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        scb.kwargs['counts'] = pkl.load(f)
    scb.make_dendrogram(no_labels=True, truncate_mode='lastp', p=50)


def cycle_worker(q):
    while True:
        job, arg = q.get()
        if type(job) is str and job == 'exit':
            return True
        job(arg)
        del arg


def get_geo_vs_crne(i):
    a, b = skyclusters.get_pair_indices(i, len(g_scb.nodes))
    A = g_scb.nodes[a]
    B = g_scb.nodes[b]
    crne = g_scb.matrix[i]
    comp = NodeComparison(A, B, g_scb.counts)
    return [crne, comp.geo_distance]


def get_ping_vs_crne(i):
    a, b = skyclusters.get_pair_indices(i, len(g_scb.nodes))
    x = g_means[a]
    y = g_means[b]
    if y < x:
        tmp = y
        y = x
        x = tmp
    crne = g_scb.matrix[i]
    return [crne, x/y]


def make_geo_vs_crne(workers=2, chunksize=500):
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder(limit=500)
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    skyclusters.gc.collect()
    pool = Pool(workers)
    global q
    q = Queue()
    dumper = Process(target=cycle_worker, args=(q,))
    dumper.start()
    data = list()
    for res in pool.imap_unordered(get_geo_vs_crne, range(len(g_scb.matrix)), chunksize):
        data.append(res)
        if len(data) >= 10000:
            q.put((dump_geo_vs_crne, data))
            del data
            data = list()
    if len(data) > 0:
        q.put((dump_geo_vs_crne, data))
    q.put(('exit', True))
    dumper.join()


def dump_pings_vs_crne(data):
    D = DataGetter()
    with open(D.fmt_path('datadir/vs_crne/pings'), 'a') as f:
        for res in data:
            f.write(json.dumps(res)+'\n')

def dump_geo_vs_crne(data):
    D = DataGetter()
    with open(D.fmt_path('datadir/vs_crne/geodists'), 'a') as f:
        for res in data:
            f.write(json.dumps(res)+'\n')

def make_pings_vs_crne(workers=2, chunksize=500):
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder(limit=500)
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    skyclusters.gc.collect()
    pings = Pings()
    global g_means
    g_means = RawArray('d', itertools.repeat(0, len(g_scb.nodes)))
    for i in range(len(g_scb.nodes)):
        node = g_scb.nodes[i]
        try:
            flat = pings.get_flat_pings(node.probe)
        except:
            flat = None
        if flat is None or len(flat) == 0:
            continue
        g_means[i] = np.mean(flat)
    del pings
    pool = Pool(workers)
    global q
    q = Queue()
    dumper = Process(target=cycle_worker, args=(q,))
    dumper.start()
    data = list()
    for res in pool.imap_unordered(get_ping_vs_crne, range(len(g_scb.nodes)), chunksize):
        data.append(res)
        if len(data) >= 10000:
            q.put((dump_pings_vs_crne, data))
            del data
            data = list()
    if len(data) > 0:
        q.put((dump_pings_vs_crne, data))
    q.put(('exit', True))
    dumper.join()

def make_homogeneity_and_completeness(workers=2, **kwargs):
    ''' also gets homogeneity '''
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder(**kwargs)
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    global g_ca
    g_ca = clusterAnalysis.ClusterAnalysis(scb=g_scb)
    skyclusters.gc.collect()
    categories = ['country', 'asn', 'prefix', 'ip24']
    thresholds = np.arange(0.05, 1.0, 0.05)
    itr = itertools.product(categories, thresholds)
    global q
    q = Queue()
    pool = Pool(workers)
    dumper = Process(target=cycle_worker, args=(q,))
    dumper.start()
    for res in pool.imap_unordered(get_homogeneity_and_completeness, itr):
        q.put((dump_homogeneity_and_completeness, res))
    q.put(('exit', True))
    dumper.join()


def make_homogeneity_and_completeness_for_resolvers(workers=2, **kwargs):
    ''' also gets homogeneity '''
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder(**kwargs)
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    '''
    2 things happening here:
        1) reducing probes to those with at least one public resolver
        2) picking the most "common" (across all probes) resolver observed by probe to label said probe
    '''
    R = skyresolvers.Resolvers()
    invR = R.get_inverse()
    keep = set()
    for resolver in invR:
        keep.update(invR[resolver])
    g_scb.nodes._probes_df = g_scb.nodes._probes_df.loc[g_scb.nodes._probes_df['probe'].isin(keep)]
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    keys = sorted(list(invR.keys()), key=lambda z: len(invR[z]))
    probes = dict()
    while len(keep):
        key = keys.pop()
        for probe in invR[key]:
            if key in keep:
                probes[probe] = key
                keep.remove(probe)
                if len(keep) == 0:
                    break
    resolvers = [probes[z] for z in g_scb.nodes._probes_df.probe.to_list()]
    g_scb.nodes._probes_df = g_scb.nodes._probes_df.assign(resolvers=resolvers)
    global g_ca
    g_ca = clusterAnalysis.ClusterAnalysis(scb=g_scb)
    skyclusters.gc.collect()
    categories = ['resolvers']
    thresholds = np.arange(0.05, 1.0, 0.05)
    itr = itertools.product(categories, thresholds)
    global q
    q = Queue()
    pool = Pool(workers)
    dumper = Process(target=cycle_worker, args=(q,))
    dumper.start()
    for res in pool.imap_unordered(get_homogeneity_and_completeness, itr):
        q.put((dump_homogeneity_and_completeness, res))
    q.put(('exit', True))
    dumper.join()


def get_homogeneity_and_completeness((category, threshold)):
    clusters = g_ca.get_clusters(threshold, method='complete')
    data = g_ca.get_homogeneity_and_completeness(clusters, category)
    data['category'] = category
    data['threshold'] = threshold
    data['nclusters'] = len(set(clusters))
    return data

def dump_homogeneity_and_completeness(data):
    D = DataGetter()
    with open(D.fmt_path('datadir/homogeneity_and_completeness/'+data['category']+'.json'), 'a') as f:
        f.write(json.dumps(data)+'\n')


if __name__ == '__main__':
    make_homogeneity_and_completeness()
