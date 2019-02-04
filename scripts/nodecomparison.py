import numpy
from geopy.distance import vincenty
from collections import defaultdict
import cPickle as pkl
from experimentdata import ExperimentData

def count_answers_across_nodes(nodes):
    count = defaultdict(lambda: 0)
    for i in range(len(nodes)):
        for key in nodes[i].results.iteritems():
            count[key] += 1
    with open(nodes.convert_path('datadir/pkls/answer_counts.pkl'), 'w') as f:
        pkl.dump(count, f)
    return count

class NodeComparison(ExperimentData):
    def __init__(self, akey, bkey, nodes, counts, **kwargs):
        self.a = nodes[akey]
        self.b = nodes[bkey]
        self.akey = akey
        self.bkey = bkey
        self.counts = counts
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def closeness(self):
        '''
        calculate and return closeness between a and b
        '''
        return -1

    @property
    def geo_distance(self):
        '''
        calculate and return geographic distance between a and b
        '''
        return -1

    @property
    def raw_performance_difference(self):
        '''
        calculate and return difference between ping results of a and b
        '''
        return -1
