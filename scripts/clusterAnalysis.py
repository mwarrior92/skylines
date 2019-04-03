from skyclusters import SkyClusterBuilder
from experimentdata import ExperimentData, DataGetter
import scipy.cluster.hierarchy as  hierarchy
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.cluster import homogeneity_score, completeness_score

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

    def get_differences(self, cluster):
        '''
        set of differing answers from a cluster and determine rarity outside of cluster
        TODO: use the existing pkls for this (get # times ans appears in cluster from cluster
        and get times ans appears overall via pkl); will be much faster than crawling all probes
        '''
        pass

    def merge_answers(self, cluster):
        '''
        TODO: this should also make a new corresponding pkl
            TODO: if we make a new pkl, will need a simple way to know which pkl to use based on
            merge iteration
        '''
        pass

