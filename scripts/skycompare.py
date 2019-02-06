from geopy.distance import vincenty
from collections import defaultdict
import cPickle as pkl
from experimentdata import ExperimentData

def count_answers_across_nodes(nodes):
    count = defaultdict(lambda: 0.0)
    for i in range(len(nodes)):
        for key in nodes[i].results.iteritems():
            count[key] += 1.0
            count[key[0]] += 1.0
    with open(nodes.fmt_path('datadir/pkls/answer_counts.pkl'), 'w') as f:
        pkl.dump(count, f)
    return count

class NodeComparison(ExperimentData):
    def __init__(self, akey, bkey, nodes, counts=None, weight_by_rarity=False, **kwargs):
        self.a = nodes[akey]
        self.b = nodes[bkey]
        self.akey = akey
        self.bkey = bkey
        self.counts = counts
        self.weight_by_rarity = weight_by_rarity
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def counts(self):
        if not hasattr(self, '_counts') or self._counts is None:
            path = self.fmt_path('datadir/pkls/answer_counts.pkl')
            with open(path, 'r') as f:
                self._counts = pkl.load(f)
        return self._counts

    @counts.setter
    def counts(self, val):
        self._counts = val

    @property
    def shared_domains(self):
        if not hasattr(self, '_shared_domains'):
            self._shared_domains = set(self.a.results.keys()).intersection(self.b.results.keys())
        return self._shared_domains

    @property
    def closeness(self):
        '''
        calculate and return closeness between a and b
        '''
        if hasattr(self, '_closeness'):
            return self._closeness
        n = list()
        for dom in self.shared_domains:
            aRes = self.a.results[dom]
            bRes = self.b.results[dom]
            if aRes == bRes:
                if self.weight_by_rarity:
                    # to how many probes did dom give same ans
                    ansCount = self.count[(dom, aRes)]
                    # how many probes tested dom
                    testCount = self.count[dom]
                    # we want rare things to weigh more
                    n.append(testCount/ansCount)
                else:
                    n.append(1.0)
        if len(self.shared_domains) == 0:
            self._closeness = -1
        else:
            self._closeness = sum(n)/float(len(self.shared_domains))
        return self._closeness

    @closeness.setter
    def closeness(self, val):
        self._closeness = val

    def get_closeness(self):
        return self.closeness

    @property
    def geo_distance(self):
        '''
        calculate and return geographic distance between a and b
        '''
        if not hasattr(self, '_geo_distance'):
            self._geo_distance = vincenty(self.a.coords, self.b.coords).km
        return self._geo_distance

    def get_geo_distance(self):
        return self.geo_distance

    @property
    def performance_difference(self):
        '''
        calculate and return difference between ping results of a and b
        '''
        return -1

if __name__ == '__main__':
    from skynodes import Nodes
    m_nodes = Nodes(limit=100)
    m_nc = NodeComparison(50,51, m_nodes)
    print(m_nc.__dict__)
    m_c = m_nc.get_closeness()
    print(m_c)
    print(m_nc.__dict__)
