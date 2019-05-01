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
from clusterAnalysis import ClusterAnalysis
import skyresolvers
import pandas
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import time
from matplotlib import cm
from collections import defaultdict
from multiprocessing import Pool
import traceback


def get_geo_vs_perf(i):
    '''
    get geo distances for a cluster
    '''
    try:
        cluster = g_clusters[i]
        geo = g_ca.get_inner_geo(cluster)
        mean_geo = g_ca.get_geo_mean(geo)
        perf = g_ca.get_inner_performance(cluster)
        mean_perf = g_ca.get_performance_mean(perf)
        out_data = list()
        perf_counts = [z[1] for z in perf.values()]
        typical = np.mean(perf_counts)
        std = np.std(perf_counts)
        for i in perf:
            if i in geo and perf[i][1] >= typical-std:
                out_data.append((i,geo[i],perf[i][0]))
    except:
        traceback.print_exception(*sys.exc_info())
        raise Exception()
    return {'raw_data': out_data, 'mean_geo': mean_geo, 'mean_perf': mean_perf}


def plot_geo_distribution(workers=2):
    '''
    pool = Pool(workers)
    data = [None]*len(g_clusters)
    for (i, res) in pool.imap_unordered(get_cluster_geo, range(len(g_clusters))):
        data[i] = res
    with open(g_ca.fmt_path('datadir/geo_dfm/threshold0.73.json'),'w') as f:
        json.dump(data,f)
    '''
    pass


def plot_cnre_distribution():
    pass


def plot_perf_distribution():
    pass


def plot_perf_vs_geo(workers=2):
    pool = Pool(workers)
    data = list()
    for res in pool.imap_unordered(get_geo_vs_perf, range(len(g_clusters))):
        data.append(res)
    with open(g_ca.fmt_path('datadir/geo_vs_perf/raw.json'), 'w') as f:
        json.dump(data,f)
    x = list()
    y = list()
    for item in data:
        _, geo, perf = zip(*item['raw_data'])
        x += geo
        y += perf
    x, y = zip(*sorted(zip(x,y), key=lambda z: z[0]))
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(x,y)
    ax.set_xlabel('mean distance (km)')
    ax.set_ylabel('mean ping (ms)')
    fig.savefig(g_ca.fmt_path('plotsdir/geo_vs_perf/raw.png'))
    plt.close(fig)


def plot_perfdfm_vs_cnredfm():
    pass


def plot_geodfm_vs_cnredfm():
    pass


if __name__ == '__main__':
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder()
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    global g_ca
    g_scb.nodes.load_pings()
    g_ca = ClusterAnalysis(scb=g_scb)
    g_clusters = g_ca.grouped_clusters(threshold=0.73)
    plot_perf_vs_geo(2)
