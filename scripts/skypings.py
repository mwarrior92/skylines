import json
from experimentdata import ExperimentData, DataGetter
import pandas
import numpy as np

class Pings(ExperimentData):
    def __init__(self, fname=None, pings=None):
        self.pings = pings
        if self.pings is None:
            self.load_pings(fname)

    def load_pings(self, fname=None):
        if fname is None:
            fname = self.fmt_path('datadir/pings/pingdata.json')
        self.pings = pandas.read_json(fname, lines=True)

    def get_pings(self, nodes, domain=None):
        try:
            if type(nodes) is not list:
                nodes = [nodes]
            if domain:
                return self.pings[(self.pings['prb'].isin(nodes)) & (self.pings['domain'] == domain)]
            else:
                return self.pings[self.pings['prb'].isin(nodes)]
        except KeyError:
            return -1.0

    def get_flat_pings(self, *args, **kwargs):
        pings = self.get_pings(*args, **kwargs)
        ret = list()
        [ret.extend(res) for res in pings['res']]
        return ret

    def get_ping_stats(self, *args, **kwargs):
        pings = self.get_flat_pings(*args, **kwargs)
        q2 = np.percentile(pings, 25)
        q4 = np.percentile(pings, 75)
        iqr = q4 - q2
        high_outlier = (1.5*iqr)+q4
        low_outlier = q2-(1.5*iqr)
        return {
                'mean': np.mean(pings),
                'std': np.std(pings),
                'high_outlier': high_outlier,
                'low_outlier': low_outlier,
                '25': q2,
                '75': q4
                }
