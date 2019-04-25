import sys
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
import pandas
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import time
from matplotlib import cm
from collections import defaultdict

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
    for category in categories:
        data = list(set(g_scb.nodes.probes_df[category]))
        with open(g_scb.fmt_path('datadir/homogeneity_and_completeness/labelsets/'+category+'.json'), 'w') as f:
            json.dump(data, f)
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
    g_scb.reduce_matrix_to_sampled()
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    keys = sorted(list(invR.keys()), key=lambda z: len(invR[z]))
    probes = dict()
    while len(keep):
        key = keys.pop()
        for probe in invR[key]:
            if probe in keep:
                probes[probe] = key
                keep.remove(probe)
                if len(keep) == 0:
                    break
    resolvers = [probes[z] for z in g_scb.nodes._probes_df.probe.to_list()]
    resolvers_set = list(set(resolvers))
    with open(g_scb.fmt_path('datadir/homogeneity_and_completeness/labelsets/resolvers.json'), 'w') as f:
        json.dump(resolvers_set, f)
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

def plot_homogeneity_and_completeness(category):
    print(category)
    D = DataGetter()
    data = pandas.read_json(D.fmt_path('datadir/homogeneity_and_completeness/'+category+'.json'), lines=True)
    data = data.sort_values(by=['threshold'])
    with open(D.fmt_path('datadir/homogeneity_and_completeness/labelsets/'+category+'.json'), 'r') as f:
        labelset = json.load(f)
    x = data.threshold.to_list()
    y1 = data.homogeneity.to_list()
    y2 = data.completeness.to_list()
    y3 = data.nclusters.to_list()
    fig, (ax) = plt.subplots(1,1)
    ax2 = ax.twinx()
    l1 = ax.plot(x,y1, '--', label='homogeneity')
    l2 = ax.plot(x,y2, label='completeness')
    l3 = ax2.plot(x,y3, 'r:', label='# clusters')
    ax2.axhline(len(labelset), color='k')
    lines = l1+l2+l3
    ax2.set_yscale('log')
    ax.set_xlabel('distance threshold')
    ax.set_ylabel('H & C')
    ax2.set_ylabel('# clusters')
    ax.legend(lines, [z.get_label() for z in lines])
    ax.set_ylim([0,1])
    fig.savefig(D.fmt_path('plotsdir/homogeneity_and_completeness/'+category+'.png'))


def get_same_diff_for_pair((category, label)):
    sys.stdout.flush()
    indices = nodes.loc[nodes[category] == label].idx.to_list()
    others = nodes.loc[nodes[category] != label].idx.to_list()
    crnes = list()
    data = {'l': label, 'c': category, 'sz': len(indices)}
    if len(indices) > 1:
        for a,b in itertools.combinations(indices, 2):
            crnes.append(1.0-scb.crne(a,b))
        data['sm'] = np.median(crnes)
        data['smn'] = np.mean(crnes)
        data['std'] = np.std(crnes)
    crnes = list()
    for a,b in itertools.product(indices, others):
        crnes.append(1.0-scb.crne(a,b))
    data['df'] = np.median(crnes)
    return data


def plot_closeness_for_category(**kwargs):
    global scb
    scb = skyclusters.SkyClusterBuilder(**kwargs)
    scb.load_matrix_from_file('datadir/matrix/matrix.json')
    global nodes
    nodes = scb.nodes.probes_df
    categories = ['asn', 'prefix', 'country', 'ip24']
    labels = [(category, set(nodes[category].tolist())) for category in categories]
    labels = [y for z in labels for y in list(itertools.product([z[0]], z[1]))]
    sames = defaultdict(list)
    diffs = defaultdict(list)
    sizes = list()
    count = 0
    fname = scb.fmt_path('datadir/closeness_vs_category/data.json')
    with open(fname, 'w') as f:
        f.write('')
    data = list()
    pool = Pool(2)
    for res in pool.imap_unordered(get_same_diff_for_pair, labels, 100):
        try:
            sames[res['c']].append(res['sm'])
            # (label, median, size)
            sizes.append((res['l'], res['smn'], res['sz'], str(res['std'])))
        except KeyError:
            pass
        diffs[res['c']].append(res['df'])
        if count % 100 == 0:
            print(res)
            sys.stdout.flush()
            with open(fname, 'a') as f:
                f.write(json.dumps(data)+'\n')
            data = list()
        count += 1
    if data:
        with open(fname, 'a') as f:
            f.write(json.dumps(data)+'\n')
        del data

    data = dict()
    for category in diffs:
        fig, (ax) = plt.subplots(1,1)
        styles = itertools.cycle(['-', '--', '-.', ':'])
        ecdf = ECDF(diffs[category])
        lines = list()
        xd, yd = list(ecdf.x), list(ecdf.y)
        lines += ax.plot(xd,yd, next(styles), label='same')
        ecdf = ECDF(sames[category])
        xs, ys = list(ecdf.x), list(ecdf.y)
        lines += ax.plot(xs,ys, next(styles), label='diff')
        m = np.percentile(diffs[category], 95)
        ax.axvline(m, color='r', linewidth=0.7)
        ax.legend(lines, [z.get_label() for z in lines])
        ax.set_xlabel('median CRNE')
        ax.set_ylabel('CDF')
        fig.savefig(scb.fmt_path('plotsdir/closeness_vs_category/'+category+'.png'))
        plt.close(fig)
        with open(scb.fmt_path('datadir/closeness_vs_category/'+category+'.json'),'w') as f:
            json.dump({'same': (xs,ys), 'diff': (xd,yd), 'm': m}, f)
    fig3, (ax3) = plt.subplots(1,1)
    _, x, y, stds = zip(*sizes)
    scatter = ax3.scatter(x, y, c=stds, edgecolors='k')
    ax3.set_xlabel('mean CRNE')
    ax3.set_ylabel('label group size')
    plt.colorbar(scatter, ticks=np.arange(0.0, 1.01, 0.2))
    fig3.savefig(scb.fmt_path('plotsdir/closeness_vs_category/size.png'))
    plt.close(fig3)
    with open(scb.fmt_path('datadir/closeness_vs_category/size.json'), 'w') as f:
        json.dump({'means': x, 'sizes': y, 'stds': stds}, f)


if __name__ == '__main__':
    '''
    make_homogeneity_and_completeness()
    make_homogeneity_and_completeness_for_resolvers()
    plot_homogeneity_and_completeness('country')
    plot_homogeneity_and_completeness('ip24')
    plot_homogeneity_and_completeness('prefix')
    plot_homogeneity_and_completeness('asn')
    plot_homogeneity_and_completeness('resolvers')
    '''
    plot_closeness_for_category()
