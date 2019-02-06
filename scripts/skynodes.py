import pandas
from experimentdata import ExperimentData

pandas.options.mode.chained_assignment = None

class Nodes(ExperimentData):
    def __init__(self, raw_mode=False, limit=None, min_tests=None, **kwargs):
        '''
        class to simplify accessing node data
        '''

        # when False, maps domains/answers to int values; this saves time on
        # comparisons (which happen a lot for closeness calculation)
        self.raw_mode = raw_mode
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def limit(self, limit):
        return len(self.probes_df)

    @limit.setter
    def limit(self, limit):
        if limit is None:
            pass
        elif type(limit) is int:
            if not self.min_tests:
                self.probes_df = self.probes_df.iloc[:limit]
            else:
                keeps = list()
                for i, n in enumerate(self.probes_df):
                    if len(n.results) >= self.min_tests:
                        keeps.append(i)
                    if len(keeps) >= limit:
                        break
                self.probes_df = self.probes_df.iloc[keeps]
        else:
            raise ValueError('expected tpye None or int for attribute "limit"')

    @property
    def min_tests(self):
        if hasattr(self, '_min_tests'):
            return self._min_tests
        return 0

    @min_tests.setter
    def min_tests(self, minTests):
        if minTests is None:
            pass
        elif type(minTests) is int:
            self._min_tests = minTests
        else:
            raise ValueError('expected type int for min_tests')

    def load_mappings(self):
        self._domi = self.load_json('datadir/mappings/dom_mapping.json')
        self._ipi = self.load_json('datadir/mappings/ip_mapping.json')
        def intdict(d):
            d2 = dict()
            for k in list(d.keys()):
                try:
                    k2 = int(k)
                except ValueError:
                    k2 = k
                if type(d[k]) == dict:
                    d2[k2] = intdict(d[k])
                else:
                    try:
                        d2[k2] = int(d[k])
                    except:
                        d2[k2] = d[k]
                del d[k]
            return d2
        self._domi = intdict(self._domi)
        self._ipi = intdict(self._ipi)

    @property
    def domi(self):
        if not hasattr(self, '_domi') or self._domi is None:
            self.load_mappings()
        return self._domi

    @property
    def ipi(self):
        if not hasattr(self, '_ipi') or self._ipi is None:
            self.load_mappings()
        return self._ipi

    def load_probes_df(self):
        self._probes_df = self.load_pkl_df('datadir/pkls/mapped_clients.pkl')

    @property
    def probes_df(self):
        if not hasattr(self, '_probes_df') or self._probes_df is None:
            self.load_probes_df()
        return self._probes_df

    @probes_df.setter
    def probes_df(self, val):
        self._probes_df = val

    def get_node(self, index):
        return self.probes_df.iloc[index]

    def get_mapped_node(self, index):
        node = self.get_node(index)
        node.results = {self.domi['i2dom'][k]: self.ipi['i2ip'][v] for (k, v) \
                in node.results.iteritems()}
        return node

    def __getitem__(self, index):
        if self.raw_mode:
            return self.get_mapped_node(index)
        else:
            return self.get_node(index)

    def __len__(self):
        return len(self.probes_df)
