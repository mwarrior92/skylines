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
from geopy.distance import geodesic as vincenty


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
    ax.set_xlabel('geo. distance from center (km)')
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


def get_domain_alignment(i):
    print('cluster '+str(i))
    c = g_clusters[i]
    if len(c) < 3:
        return None
    return g_ca.get_domain_alignment(c)


def plot_domain_alignment():
    all_sets = list()
    pool = Pool(3)
    for tmp_counts in pool.imap_unordered(get_domain_alignment, range(len(g_clusters))):
        if tmp_counts:
            all_sets.append(tmp_counts)
    try:
        with open(g_ca.fmt_path('datadir/domain_alignment/raw.json'),'w') as f:
            json.dump(all_sets,f)
    except:
        print('failed to save raw')
    data = list()
    means = dict()
    for i, cluster in enumerate(all_sets):
        alns, sizes, perfs = zip(*cluster.values())
        mean_aln = np.mean(alns)
        perfs = [z for z in perfs if z > 0]
        if perfs:
            mean_perf = np.mean(perfs)
        else:
            mean_perf = None
        means[i] = (mean_aln, mean_perf)
        for dom, val in cluster.items():
            aln, s, perf = val
            data.append((dom, aln - mean_aln,
                perf - mean_perf if mean_perf and perf else None))
    with open(g_ca.fmt_path('datadir/domain_alignment/deviations.json'),'w') as f:
        json.dump(data,f)
    D = DataGetter()
    with open(D.fmt_path('datadir/domain_alignment/deviations.json'),'r') as f:
        data = json.load(f)
    doms, aln_devs, perf_devs = zip(*data)
    fig, ax = plt.subplots(figsize=(6,3.5))
    ecdf = ECDF(aln_devs)
    ax.plot(list(ecdf.x), list(ecdf.y))
    ax.set_xlabel('distance from mean alignment')
    ax.set_ylabel('CDF')
    fig.savefig(D.fmt_path('plotsdir/domain_alignment/alignment.png'))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    aln_devs, perf_devs = zip(*[z for z in zip(aln_devs, perf_devs) if z[1]])
    heatmap, x, y = np.histogram2d(aln_devs,perf_devs,bins=50)
    extent = [x[0], x[-1], y[0], y[-1]]
    pos = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Greys')
    fig.colorbar(pos)
    ax.set_xlabel('distance from mean alignment')
    ax.set_ylabel('distance from mean performance')
    fig.savefig(D.fmt_path('plotsdir/domain_alignment/align_vs_perf.png'))
    plt.close(fig)




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


def get_the_center(i):
    try:
        return i, g_ca.get_dists_from_geo_center(g_clusters[i])[:2]
    except Exception as e:
        print(e)
        return i, None


def sort_the_cneters(i):
    cluster = g_clusters[i]
    coords = dict()
    for z in cluster:
        tmp = g_ca.scb.nodes[z].coords
        if tmp:
            coords[z] = tuple(reversed(tmp[0]))
    out = list()
    for client, crd in coords.items():
        try:
            dists = list()
            cid, ctr = g_centers[i]
            if client == cid:
                continue
            di = vincenty(crd, ctr).km
            cnrei = g_ca.scb.cnre(client, cid)
            for j, (cid, ctr) in g_centers.items():
                if i != j:
                    d = vincenty(crd, ctr).km
                    cnre = g_ca.scb.cnre(client, cid)
                    dists.append((j,cnre,d))
            dists = sorted(dists, key=lambda z: z[-1])
            out.append((client, i, cnrei, di, dists))
        except Exception as e:
            print(str(crd)+', '+str(g_centers[i]))
            sys.stdout.flush()
            traceback.print_exception(*sys.exc_info())
            raise e
    return out

def get_nearest_centers(workers):
    global g_centers
    g_centers = dict()
    pool = Pool(workers)
    for i, center in pool.imap_unordered(get_the_center, g_clusters.keys()):
        if center:
            g_centers[i] = center
    pool.terminate()
    pool = Pool(workers)
    data = list()
    print('len: '+str(len(g_centers)))
    progress = 0
    for out in pool.imap_unordered(sort_the_cneters, g_centers.keys()):
        data += out
        progress += 1
        print(progress)
        sys.stdout.flush()
    with open(g_ca.fmt_path('datadir/all_center_dists.json'),'w') as f:
        json.dump(data, f)
    tops = list()
    for item in data:
        client, group, ctr_cnre, ctr_dist, dists = item
        closest_ctr_grp, closest_ctr_cnre, closest_ctr_dist = dists[0]
        tops.append((group, closest_ctr_grp, ctr_cnre, closest_ctr_cnre,
            ctr_dist, closest_ctr_dist))
    with open(g_ca.fmt_path('datadir/nearest_centers.json'),'w') as f:
        json.dump(tops,f)


def plot_nearest_centers():
    D = DataGetter()
    with open(D.fmt_path('datadir/nearest_centers.json'),'r') as f:
        data = json.load(f)

    cnres = list()
    dists = list()
    for item in data:
        _, _, c0, c1, d0, d1 = item
        cnres.append((1.0-c0, 1.0-c1))
        dists.append((d0, d1))
    x,y = zip(*cnres)
    #rng = [minlim, maxlim]
    #heatmap, x, y = np.histogram2d(x,y,bins=100, range=[rng,rng])
    #extent = [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(4,6))
    ax.scatter(x,y, alpha=0.1)
    #minlim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    #maxlim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #pos = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Greys')
    #fig.colorbar(pos)
    ax.set_xlabel('CNRE with default center')
    ax.set_ylabel('CNRE with closest center')
    ax.plot([0,1],[0, 1], 'r')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(D.fmt_path('plotsdir/nearest_centers_cnre.png'))
    plt.close(fig)
    x,y = zip(*dists)
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.scatter(x,y, alpha=0.1)
    ax.set_xlabel('km to default center')
    ax.set_ylabel('km to closest center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    minlim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    maxlim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.set_xlim([minlim,maxlim])
    ax.set_ylim([minlim,maxlim])
    ax.plot([minlim,maxlim],[minlim,maxlim], 'r')
    fig.savefig(D.fmt_path('plotsdir/nearest_centers_dist.png'))
    plt.close(fig)


if __name__ == '__main__':
    global g_scb
    g_scb = skyclusters.SkyClusterBuilder()
    g_scb.load_matrix_from_file('datadir/matrix/matrix.json')
    with open(g_scb.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
        g_scb.kwargs['counts'] = pkl.load(f)
    global g_ca
    #g_scb.nodes.load_pings()
    g_ca = ClusterAnalysis(scb=g_scb)
    g_clusters = g_ca.grouped_clusters(threshold=1.0-0.73)
    '''
    #with open(g_ca.fmt_path('datadir/g_clusters.json'),'w') as f:
    #    json.dump([list(z) for z in g_clusters],f)
    #plot_perf_vs_geo(2)
    #plot_geo_mean(2)
    #plot_geo_centers()
    with open(g_ca.fmt_path('datadir/nclusters.txt'),'w') as f:
        f.write(str([len(g_clusters), np.median([len(z) for z in g_clusters])]))
    '''
    g_ca.scb.nodes.keep_only([])
    g_ca.scb.nodes.attach_pings()
    plot_domain_alignment()
    '''
    g_ca.scb.nodes.keep_only('coords')
    with open(g_scb.fmt_path('datadir/g_clusters.json'),'r') as f:
        g_clusters = json.load(f)
    g_clusters = {i: g_clusters[i] for i in range(len(g_clusters)) if len(g_clusters[i]) > 2}
    get_nearest_centers(2)
    plot_nearest_centers()
    plot_domain_alignment()
    '''
