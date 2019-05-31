import sys
from skyclusters import SkyClusterBuilder
from skycompare import NodeComparison
from experimentdata import ExperimentData, DataGetter
import scipy.cluster.hierarchy as  hierarchy
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import itertools
import cPickle as pkl
#from geopy.distance import vincenty
from geopy.distance import geodesic as vincenty

class ClusterAnalysis(ExperimentData):
    def __init__(self, matrix_file=None, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

        if not hasattr(self, 'scb'):
            self.scb = SkyClusterBuilder(**kwargs)
            if not hasattr(self.scb, 'matrix'):
                with open(matrix_file, 'r') as f:
                    self.scb.matrix = np.array(json.load(f))
            if type(self.scb.matrix) is not np.ndarray:
                self.scb.matrix = np.array(self.scb.matrix)

    def get_ipairs_by_distance(self, dist, d2=None):
        '''
        return pairs of probes and their distance
        NOTE: pairs are in terms of their indices in scb.nodes
        '''
        pairs = list()
        if d2 is None:
            inds = np.nonzero(self.scb.matrix == dist)
        else:
            inds = np.where(dist <= self.scb.matrix <= d2)
        for i in inds:
            pairs.append((self.scb.get_pair_indices(i), self.scb.matrix[i]))
        return pairs

    def get_ppairs_by_distance(self, dist, d2=None):
        '''
        return pairs of probes and their distance
        NOTE: pairs are tuples of dataframes corresponding to probes
        '''
        pairs = self.get_ipairs_by_distance(dist, d2)
        ppairs = list()
        for pair in pairs:
            ppairs.append(((self.scb.nodes[pair[0][0]], self.scb.nodes[pair[0][1]]),
                pair[1]))
        return ppairs

    def linkage(self, method='complete', **kwargs):
        m = 'linkage_'+method
        if not hasattr(self, m):
            setattr(self, m, hierarchy.linkage(self.scb.matrix, method=method))
        return getattr(self, m)

    def get_clusters(self, threshold, **kwargs):
        print('getting clusters')
        L = self.linkage(**kwargs)
        data = hierarchy.fcluster(L, threshold, 'distance')
        return data

    def grouped_clusters(self, data=None, threshold=None, **kwargs):
        if data:
            return [np.nonzero(data == z)[0] for z in set(data)]
        else:
            data = self.get_clusters(threshold, **kwargs)
            return [np.nonzero(data == z)[0] for z in set(data)]

    def get_homogeneity_and_completeness(self, clusters, category):
        labels = getattr(self.scb.nodes, 'to_'+category)(range(len(self.scb.nodes)))
        keys = dict()
        for i, label in enumerate(labels):
            if label not in keys:
                keys[label] = len(keys)
            labels[i] = keys[label]
        hs = homogeneity_score(labels, clusters)
        cs = completeness_score(labels, clusters)
        return {'homogeneity': hs, 'completeness': cs}

    def get_addr_differences(self, cluster):
        '''
        set of differing answers from a cluster and determine rarity outside of cluster
        TODO: make 'flat_answer_counts.pkl', which groups by addr instead of by (site,addr)
        '''
        answers = defaultdict(set)
        for i in cluster:
            for site, addrs in self.scb.nodes[i].results:
                for addr in addrs:
                    answers[(site, addr)].add(i)
        D = DataGetter()
        with open(D.fmt_path('datadir/pkls/flat_answer_counts.pkl'), 'r') as f:
            global_counts = pkl.load(f)
        uniques = defaultdict(list)
        for site, addr in answers:
            world = float(global_counts[(site,addr)])
            coverage = float(len(answers[(site,addr)])) / world
            if coverage >= 0.90:
                uniques[site].append(addr)
        return uniques

    def compare_mergeables(self, cluster, mergeables):
        listpings = self.scb.nodes.to_ping(cluster)
        pings = defaultdict(lambda: defaultdict(list))
        for i, client in enumerate(listpings):
            res = self.scb.nodes[cluster[i]].results
            for site in client:
                pings[site][res[site]].append(client[site])
        comps = dict()
        for site in mergeables:
            if site in pings:
                comps[site] = list()

    def merge_answers(self, cluster, mergeables):
        '''
        TODO: this should also make a new corresponding pkl
            TODO: if we make a new pkl, will need a simple way to know which pkl to use based on
            merge iteration
        '''
        pass

    def get_inner_cnres(self, cluster):
        cnres = defaultdict(list)
        for i, j in itertools.combinations(cluster, 2):
            cnre = self.scb.cnre(i,j)
            cnres[i].append(cnre)
            cnres[j].append(cnre)
        for i in cnres:
            cnres[i] = np.mean(cnres[i])
        return cnres

    def get_inner_geo(self, cluster):
        dists = defaultdict(list)
        nodes = list()
        for z in cluster:
            coords = self.scb.nodes[z].coords
            if not coords or not coords[0]:
                continue
            else:
                coords = coords[0][1], coords[0][0]
            nodes.append((z,coords))
        nodes = [(z[0],z[1][0]) for z in nodes if z[1]]
        for (i,a),(j,b) in itertools.combinations(nodes, 2):
            d = vincenty(a, b).km
            dists[i].append(d)
            dists[j].append(d)
        for i in dists:
            dists[i] = np.mean(dists[i])
        return dict(dists)

    def get_inner_geo_list(self, cluster):
        nodes = list()
        for z in cluster:
            coords = self.scb.nodes[z].coords
            if not coords or not coords[0]:
                continue
            else:
                coords = coords[0][1], coords[0][0]
            nodes.append((z,coords))
        nodes = [(z[0],z[1][0]) for z in nodes if z[1]]
        dists = list()
        for (i,a),(j,b) in itertools.combinations(nodes, 2):
            d = vincenty(a, b).km
            dists.append(d)
        return dists

    def get_inner_performance(self, cluster):
        nodes = [(z,self.scb.nodes[z].pings) for z in cluster]
        nodes = {z[0]:(z[1][0], z[1][2]) for z in nodes if z[1]}
        return nodes

    def get_performance_mean(self, inner):
        counts = [inner[z][1] for z in inner]
        typical = np.mean(counts)
        std = np.std(counts)
        nodes = list()
        for i in inner:
            p,c = inner[i]
            if float(c) >= typical-std:
                nodes.append((i, p))
        mean = np.mean([z[1] for z in nodes])
        return mean

    def get_geo_mean(self, inner):
        mean = np.mean(list(inner.values()))
        return mean

    def get_geo_center(self, inner):
        mean = np.mean(list(inner.values()))
        center = sorted(list(inner.keys()), key=lambda z: abs(inner[z]-mean))[0]
        return self.scb.nodes[center]

    def get_dists_from_geo_center(self, cluster):
        dists = defaultdict(list)
        nodes = list()
        for z in cluster:
            coords = self.scb.nodes[z].coords
            if not coords or not coords[0]:
                continue
            else:
                coords = coords[0][1], coords[0][0]
            nodes.append((z,coords))
        for (i,a),(j,b) in itertools.combinations(nodes, 2):
            d = vincenty(a, b).km
            dists[i].append(d)
            dists[j].append(d)
        for i in dists:
            dists[i] = sum(dists[i])
        center = sorted(list(dists.keys()), key=lambda z: dists[z])[0]
        center_loc = self.scb.nodes[center].coords[0]
        center_loc = [center_loc[1], center_loc[0]]
        return (center, center_loc, {z[0]: vincenty(z[1], center_loc).km for z in nodes})

    def get_cnre_mean(self, inner):
        mean = np.mean(list(inner.values()))
        return mean

    def get_domain_alignment(self,cluster):
        '''
        how much did site answers match for a given cluster
        TODO: maybe we can use homogeneity to do this?
        '''
        answers = defaultdict(set)
        tests = defaultdict(int)
        for i in cluster:
            for site, addrs in self.scb.nodes[i].results.items():
                for addr in addrs:
                    tests[site] += 1
                    answers[site].add(addr)
        counts = dict()
        D = DataGetter()
        for i in tests:
            t = float(tests[i])
            if t == 0:
                continue
            a = float(len(answers[i]))
            site = D.int2dom(int(i))
            pings = self.scb.nodes.get_pings_for_domain(cluster,site)
            print(site)
            print(pings)
            if len(pings):
                pings = [np.mean(z) for z in pings.res.to_list() if z]
                pings = np.median(pings)
            else:
                pings = -1
            site = self.scb.nodes.get_raw_domain(i)
            total = self.scb.kwargs['counts'][i]
            counts[site] = (a/t, total, pings)
        return counts
