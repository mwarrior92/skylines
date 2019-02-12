import pandas
from experimentdata import ExperimentData
from collections import defaultdict
from scipy.stats import mode
from random import sample

pandas.options.mode.chained_assignment = None

class CollapsedNode(ExperimentData):
    '''
    class to avoid running into over-complicated iteration issues when grouped
    by probe (because each column entry becomes a list)
    '''
    def __init__(self, node, **kwargs):
        # make sure we're not being redundant
        if type(node) is CollapsedNode:
            return node
        elif type(node) is not pandas.core.series.Series:
            raise ValueError('! node must be of type pandas Series')
        self.node = node
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __getattr__(self, k):
        if k in self.node and k not in {'local', 'coords',
                'rcvd', 'entry_id'} and type(getattr(self.node, k)) is list:
            # if it's a column, just return most common value
            m, c = mode(self.node[k])
            setattr(self, k, m[0])
            return m[0]
        elif hasattr(self.node, k):
            return getattr(self.node, k)
        else:
            raise AttributeError('! no attribute '+k)

    @property
    def results(self):
        if hasattr(self, '_results'):
            pass
        elif type(self.node.results) is list:
            self._results = defaultdict(set)
            for res in self.node.results:
                for k, v in res.iteritems():
                    self._results[k].add(v)
        else:
            return self.node.results
        return self._results


class Nodes(ExperimentData):
    def __init__(self, group_mode='probe', raw_mode=False, limit=None, min_tests=None, **kwargs):
        '''
        class to simplify accessing node data
        '''

        self.rules_applied = False
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.raw_mode = raw_mode
        self.min_tests = min_tests
        self.limit = limit
        self.group_mode = group_mode
        self.apply_rules()

    def __getattr__(self, k):
        if hasattr(self, '_probes_df') and hasattr(self._probes_df, k):
            return getattr(self.probes_df, k)
        else:
            raise AttributeError('! no member '+k)

    def apply_rules(self):
        print('applying rules')
        self.rules_applied = True
        getattr(self, 'group_by_'+self.group_mode)()
        if not self.min_tests:
            self._probes_df = self._probes_df.iloc[:self.limit]
            keeps = range(len(self._probes_df))
        else:
            keeps = list()
            for i, n in self._probes_df.iterrows():
                # make sure we're dealing with 1D data
                n = CollapsedNode(n)
                if len(n.results) >= self.min_tests:
                    keeps.append(i)
        self._probes_df = self._probes_df.iloc[keeps]
        if self.limit:
            keeps = range(len(self._probes_df))
            if self._sample == 'random':
                keeps = sample(keeps, self.limit)
            else:
                keeps = keeps[:self.limit]
            self._probes_df = self._probes_df.iloc[keeps]

    def group_by_probe(self):
        print('grouping by probe')
        self.load_probes_df()
        groups = self.probes_df.groupby('probe', as_index=False)
        self.probes_df = groups.aggregate(lambda z: list(z))

    def group_by_src_addr(self):
        self.load_probes_df()

    @property
    def group_mode(self):
        if not hasattr(self, '_group_mode'):
            self._group_mode = None
        return self._group_mode

    @group_mode.setter
    def group_mode(self, group_mode):
        if group_mode in {'probe', 'src_addr'}:
            self._group_mode = group_mode
        elif group_mode is None:
            self._group_mode = 'probe'
        else:
            raise ValueError('expected "probe" or "src_addr" for group_mode')
        self.rules_applied = False

    @property
    def limit(self):
        if not hasattr(self, '_limit'):
            self._limit = None
        if not hasattr(self, '_sample'):
            self._sample = 'random'
        return self._limit

    @limit.setter
    def limit(self, limit):
        if not limit:
            self._limit = None
        elif type(limit) is int:
            self._limit = limit
        else:
            raise ValueError('expected tpye None or int for attribute "limit"')
        self.rules_applied = False

    @property
    def min_tests(self):
        if hasattr(self, '_min_tests'):
            return self._min_tests
        return 0

    @min_tests.setter
    def min_tests(self, minTests):
        if type(minTests) is int:
            self._min_tests = minTests
        elif minTests is None:
            self._min_tests = 0
        else:
            raise ValueError('expected type int for min_tests')
        self.rules_applied = False

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
        ''' dict for mapping btwn domain and int '''
        if not hasattr(self, '_domi') or self._domi is None:
            self.load_mappings()
        return self._domi

    @property
    def ipi(self):
        ''' dict for mapping between answer IP and int '''
        if not hasattr(self, '_ipi') or self._ipi is None:
            self.load_mappings()
        return self._ipi

    def load_probes_df(self):
        print('loading probes...')
        self._probes_df = self.load_pkl_df('datadir/pkls/mapped_clients.pkl')

    @property
    def probes_df(self):
        if not self.rules_applied:
            self.apply_rules()
        if not hasattr(self, '_probes_df') or self._probes_df is None:
            self.load_probes_df()
        return self._probes_df

    @probes_df.setter
    def probes_df(self, val):
        self._probes_df = val

    def get_node(self, index):
        return self.probes_df.iloc[index]

    def get_mapped_node(self, index):
        node = CollapsedNode(self.get_node(index))
        node.results = {self.domi['i2dom'][k]: self.ipi['i2ip'][v] for (k, v) \
                in node.results.iteritems()}
        return node

    def __getitem__(self, index):
        if self.raw_mode:
            # NOTE: this is a CollapsedNode
            return self.get_mapped_node(index)
        else:
            # NOTE: this is a Series
            return self.get_node(index)

    def __len__(self):
        return len(self.probes_df)

