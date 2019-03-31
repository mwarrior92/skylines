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

def make_completeness(workers=2, **kwargs):
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder(**kwargs)
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    g_ca = clusterAnalysis.ClusterAnalysis(scb=g_scb)
    skyclusters.gc.collect()
    categories = ['country', 'asn', 'prefix', 'ip24', 'resolvers']
    thresholds = np.arange(0.1, 1.0, 0.1)
    itr = itertools.product(categories, thresholds)
    global q
    q = Queue()
    dumper = Process(target=cycle_worker, args=(q,))
    dumper.start()
    for category, threshold, data in pool.imap_unordered(get_completeness, itr):
        q.put(('dump_completeness', (category, data)))
    q.put(('exit', True))
    dumper.join()


def get_completeness((category, threshold, **kwargs)):
    clusters = g_ca.grouped_clusters(threshold, **kwargs)
    return g_ca.get_homogeneity_and_completeness(clusters, category)


def dump_completeness((category, threshold, data)):
    D = DataGetter()
    with open(D.fmt_path('datadir/completeness/'+category+'.json'), 'a') as f:
        f.write(json.dumps([threshold, data])+'\n')

