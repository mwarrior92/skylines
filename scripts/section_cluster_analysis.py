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
import traceback
import geopandas as gp
from descartes import PolygonPatch
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter, LogFormatter
from shapely.geometry import Point, Polygon


def get_geo_vs_perf(i):
    '''
    get geo distances for a cluster
    '''
    try:
        cluster = g_clusters[i]
        if len(cluster) < 3:
            return {}
        try:
            geocenter, geocenterloc, geo = g_ca.get_dists_from_geo_center(cluster)
        except:
            print(cluster)
            return {}
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
    return {'raw_data': out_data, 'geo_center': geocenter, 'geo_center_loc': geocenterloc,
            'mean_perf': mean_perf}


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
        if res:
            data.append(res)
    x = list()
    y = list()
    with open(g_ca.fmt_path('datadir/geo_vs_perf/raw.json'), 'w') as f:
        json.dump(data,f)
    for item in data:
        _, geo, perf = zip(*item['raw_data'])
        x += geo
        y += perf
    x, y = zip(*sorted(zip(x,y), key=lambda z: z[0]))
    slope, offset = np.polyfit(x,y,1)
    with open(g_ca.fmt_path('datadir/geo_vs_perf/plot.json'), 'w') as f:
        json.dump({'x': x, 'y': y, 'slope': slope, 'offset': offset},f)
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.scatter(x,y)
    ax.plot(x, [offset + slope*z for z in x], 'r')
    ax.set_xlabel('mean distance (km)')
    ax.set_ylabel('mean ping (ms)')
    fig.savefig(g_ca.fmt_path('plotsdir/geo_vs_perf/raw.png'))
    plt.close(fig)


def plot_geo_centers():
    D = DataGetter()
    world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    with open(D.fmt_path('datadir/geo_vs_perf/raw.json'), 'r') as f:
        geometry, sizes = zip(*[(Point(z['geo_center_loc'][1],z['geo_center_loc'][0]), len(z['raw_data'])) for z in json.load(f)])
    fig, ax = plt.subplots(figsize=(15,15))
    world.plot(edgecolor='gray', ax=ax)
    gdf = gp.GeoDataFrame(geometry=list(geometry))
    gdf.plot(ax=ax, markersize=5, color='yellow')
    fig.savefig(D.fmt_path('plotsdir/geo_centers.png'))
    plt.close(fig)



def plot_perfdfm_vs_cnredfm():
    pass


def plot_geodfm_vs_cnredfm():
    pass


def get_domain_alignment((i,c)):
    print('cluster '+str(i))
    return g_ca.get_domain_alignment(c)


def plot_domain_alignment():
    counts = defaultdict(list)
    all_sets = list()
    pool = Pool()
    for tmp_counts in pool.imap_unordered(get_domain_alignment, enumerate(g_clusters)):
        all_sets.append(tmp_counts)
    try:
        with open(g_ca.fmt_path('datadir/domain_alignment/raw.json'),'w') as f:
            json.dump(all_sets,f)
    except:
        print('failed to save raw')
    with open(g_ca.fmt_path('datadir/domain_alignment/per_cluster.json'),'w') as f:
        for counts in all_sets:
            data = sorted([[k]+list(counts[k]) for k in counts], key=lambda z: z[1])[:4]
            json.dump(data,f)
            f.write('\n')
            f.write('---------------------------------------------------------\n\n')


def get_dist_list(i):
    cluster = g_clusters[i]
    if len(cluster) < 3:
        return []
    try:
        return g_ca.get_inner_geo_list(cluster)
    except:
        return []


def plot_geo_mean(workers):
    pool = Pool(workers)
    stats = list()
    for dists in pool.imap_unordered(get_dist_list, range(len(g_clusters))):
        if dists:
            std = np.std(dists, ddof=1)
            diameter = float(max(dists))
            u = np.mean(dists)
            length = len(dists)
            stats.append((std,diameter,u, length))
    with open(g_ca.fmt_path('datadir/geo_stds/raw.json'),'w') as f:
        json.dump(stats,f)
    fig, ax = plt.subplots(figsize=(6,3.5))
    stds, diameters, us, length = zip(*stats)
    ecdf = ECDF(us)
    ax.plot(list(ecdf.x), list(ecdf.y))
    m = np.median(us)
    ax.axvline(m, color='r', linestyle='--')
    ax.text(m, 0.85, str(m).split('.')[0], ha='center', va='center', backgroundcolor='white')
    ax.set_xlabel('mean geographic distance (km)')
    ax.set_ylabel('CDF')
    fig.savefig(g_ca.fmt_path('plotsdir/geo_means/geo_means.png'))
    plt.close(fig)



if __name__ == '__main__':
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder()
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    global g_ca
    g_scb.nodes.load_pings()
    g_ca = ClusterAnalysis(scb=g_scb)
    g_clusters = g_ca.grouped_clusters(threshold=1.0-0.73)
    plot_perf_vs_geo(2)
    plot_geo_mean(2)
    #plot_geo_centers()
    '''
    with open(g_ca.fmt_path('datadir/nclusters.txt'),'w') as f:
        f.write(str([len(g_clusters), np.median([len(z) for z in g_clusters])]))
    '''
    #g_scb.nodes.attach_pings()
    #plot_domain_alignment()
