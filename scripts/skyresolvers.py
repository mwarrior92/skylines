from experimentdata import ExperimentData, DataGetter
import json
from collections import defaultdict
from IPy import IP

class Resolvers(ExperimentData):
    def __init__(self, fname=None, inverse=None):
        if fname is None:
            fname = self.fmt_path('datadir/resolvers/all_resolvers.json')
        with open(fname, 'r') as f:
            self.resolvers = {int(k):set(v) for (k,v) in json.load(f).items()}

        if inverse is None:
            inverse = self.fmt_path('datadir/resolvers/inverse.json')
        try:
            with open(inverse, 'r') as f:
                self.inverse = json.load(f)
        except:
            self.inverse = self.get_inverse()

    def __getitem__(self, key):
        try:
            if type(key) is int:
                return self.resolvers[key]
            elif type(key) in (set, list, tuple):
                return [(k, v) for (k, v) in self.resolvers if set(key) == v]
            else:
                return [(k, v) for (k, v) in self.resolvers if key in v]
        except KeyError:
            return []

    def get_inverse(self):
        ''' resolver: probe ids '''
        inverse = defaultdict(list)
        for probe, resolvers in self.resolvers.items():
            for resolver in resolvers:
                iptype = IP(resolver).iptype()
                if not (iptype.startswith('PUBLIC') or iptype.startswith('ALLOCATED')):
                    # make sure it's a public address
                    continue
                inverse[resolver].append(probe)
        return inverse
