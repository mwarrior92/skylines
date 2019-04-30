from skyclusters import SkyClusterBuilder
from experimentdata import ExperimentData, DataGetter
import scipy.cluster.hierarchy as  hierarchy
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import itertools
import cPickle as pkl
from geopy.distance import vincenty

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

    def get_per_site_differences(self, cluster):
        '''
        set of differing answers from a cluster and determine rarity outside of cluster
        TODO: use the existing pkls for this (get # times ans appears in cluster from cluster
        and get times ans appears overall via pkl); will be much faster than crawling all probes
        '''
        answers = defaultdict(lambda: defaultdict(set))
        for i in cluster:
            for site, addrs in self.scb.nodes[i].results:
                for addr in addrs:
                    answers[site][addr].add(i)
        D = DataGetter()
        with open(D.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
            global_counts = pkl.load(f)
        mergeables = dict()
        for site in answers:
            if len(answers[site]) > 1:
                uniques = list()
                for addr in answers[site]:
                    world = float(global_counts[(site,addr)])
                    coverage = float(len(answers[site][addr])) / world
                    if coverage >= 0.90:
                        uniques.append((addr, coverage, answers[site][addr]))
                if len(uniques) > 1:
                    mergeables[site] = uniques
        return mergeables

    def get_addr_differences(self, cluster):
        '''
        set of differing answers from a cluster and determine rarity outside of cluster
        TODO: make 'flat_answer_counts.pkl', which groups by addr instead of by (site,addr)
        '''
        answers = defaultdict(set)
        for i in cluster:
            for site, addrs in self.scb.nodes[i].results:
                for addr in addrs:
                    answers[addr].add(i)
        D = DataGetter()
        with open(D.fmt_path('datadir/pkls/flat_answer_counts.pkl'), 'r') as f:
            global_counts = pkl.load(f)
        uniques = list()
        for addr in answers:
            world = float(global_counts[(site,addr)])
            coverage = float(len(answers[site][addr])) / world
            if coverage >= 0.90:
                uniques.append((addr, coverage, answers[site][addr]))
        return uniques

    def merge_answers(self, cluster):
        '''
        TODO: this should also make a new corresponding pkl
            TODO: if we make a new pkl, will need a simple way to know which pkl to use based on
            merge iteration
        '''
        pass

    def get_inner_cnres(self, cluster):
        cnres = defaultdict(list)
        for i, j in itertools.combinations(cluster):
            cnre = self.scb.cnre(i,j)
            cnres[i].append(cnre)
            cnres[j].append(cnre)
        for i in cnres:
            cnres[i] = np.mean(cnres[i])
        return cnres

    def get_inner_geo(self, cluster):
        dists = defaultdict(list)
        nodes = [(z,self.scb.nodes[z].coords) for z in cluster]
        nodes = [(z[0],z[1][0]) for z in nodes if z[1]]
        for (i,a),(j,b) in itertools.combinations(nodes):
            d = vincenty(a, b).km
            dists[i].append(d)
            dists[j].append(d)
        for i in dists:
            dists[i] = np.mean(dists[i])
        return dists

    def get_inner_performance(self, cluster):
        nodes = [(z,self.scb.nodes[z].pings) for z in cluster]
        nodes = [(z[0],z[1][0], z[1][2]) for z in nodes if z[1]]
        return nodes

    def get_performance_dfm(self, inner):
        # dfm = distance from mean
        counts = [z[2] for z in inner]
        typical = np.mean(counts)
        std = np.std(counts)
        nodes = list()
        for i,p,c in inner:
            if float(c) >= typical-std:
                nodes.append((i, p))
        mean = np.mean([z[1] for z in nodes])
        return (mean, [(z[0], z[1] - mean) for z in nodes])

    def get_geo_dfm(self, inner):
        # dfm = distance from mean
        mean = np.mean(list(inner.values()))
        return (mean, [(z,inner[z]-mean) for z in inner])

    def get_cnre_dfm(self, inner):
        mean = np.mean(list(inner.values()))
        return (mean, [(z,inner[z]-mean) for z in inner])

    def get_domain_alignment(self,cluster):
        '''
        how much did site answers match for a given cluster
        TODO: maybe we can use homogeneity to do this?
        '''
        answers = defaultdict(lambda: defaultdict(set))
        tests = defaultdict(int)
        for i in cluster:
            for site, addrs in self.scb.nodes[i].results:
                for addr in addrs:
                    tests[site] += 1
                    answers[site].add(addr)
        counts = dict()
        for site in tests:
            t = float(len(tests[site]))
            a = float(len(answers[site]))
            counts[site] = a/t
        return counts
