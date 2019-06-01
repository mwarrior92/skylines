import pandas
from experimentdata import ExperimentData, DataGetter
from collections import defaultdict
from scipy.stats import mode
from random import sample
from numpy import array, ndarray, zeros
import gc
import skyresolvers
import skypings

pandas.options.mode.chained_assignment = None

def CollapsedNode(node):
    '''
    func to avoid running into over-complicated iteration issues when grouped
    by probe (because each column entry becomes a list)
    '''
    results = defaultdict(set)
    for res in node.results:
        for k, v in res.iteritems():
            results[k].add(v)
    node['results'] = dict(results)
    for k in node.index:
        if k not in {'local', 'coords', 'rcvd',
                'entry_id'} and type(node[k]) is list:
            # if it's a column, just return most common value
            m, c = mode(node[k])
            node[k] = m[0]
    return node

def ChangePrefix(node, newp, oldp=24):
    results = defaultdict(set)
    d = DataGetter()
    for k, v in node.results.iteritems():
        v = d.int2ip(v, oldp)
        v = d.ip2int(v, newp)
        results[k] = v
    node['results'] = dict(results)
    return node


class Nodes(ExperimentData):
    def __init__(self, group_mode='probe', raw_mode=False, limit=0,
            min_tests=0, **kwargs):
        '''
        class to simplify accessing node data
        '''
        self.rules_applied = False
        self.raw_mode = raw_mode
        self.min_tests = min_tests
        self.limit = limit
        self.group_mode = group_mode
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if 'from_file' in kwargs and kwargs['from_file']:
            path = self.fmt_path('objectdir/probes_df/'+self.timeid+'.json')
            self._probes_df = pandas.read_json(path)
        if 'rules_applied' not in kwargs or not kwargs['rules_applied']:
            self.apply_rules()

    def __getattr__(self, k):
        if hasattr(self, '_probes_df') and hasattr(self._probes_df, k):
            return getattr(self.probes_df, k)
        elif k.startswith('to_'):
            return self.to_(k.split('to_')[1])
        else:
            raise AttributeError(str(type(self))+', '+k)

    def save_self(self, timeid=None):
        if timeid is None:
            timeid = self.timeid
        if hasattr(self, '_probes_df') and len(self._probes_df) > 0:
            path = self.fmt_path('objectdir/probes_df/'+timeid+'.json')
            self._probes_df.to_json(path)
        super(type(self),self).save_self(timeid)

    def load_resolvers(self):
        r = skyresolvers.Resolvers()
        self._probes_df = self._probes_df.assign(resolvers=[list(r[z]) for z \
                in self._probes_df.probe])

    def load_pings(self):
        p = skypings.Pings()
        self._probes_df = self._probes_df.assign(pings=[p.get_ping_stats(z) for z \
                in self._probes_df.probe])

    def load_ping_means(self):
        p = skypings.Pings()
        self._probes_df = self._probes_df.assign(pings=[p.get_ping_mean(z) for z \
                in self._probes_df.probe])

    def attach_pings(self):
        self.pingdata = skypings.Pings(self.fmt_path('datadir/pings/light_pings.json'))

    def get_pings_for_domain(self, nodes, domain):
        if not hasattr(self, 'pingdata'):
            self.attach_pings()
        return self.pingdata.get_pings(nodes, domain)

    def keep_only(self, fields):
        fields = set(fields)
        fields.add('idx')
        fields.add('probe')
        drops = [z for z in self._probes_df.columns.to_list() if z not in fields]
        self._probes_df.drop(columns=drops)
        gc.collect()

    def apply_rules(self):
        print('applying rules')
        self.rules_applied = True
        getattr(self, 'group_by_'+self.group_mode)()
        if self.min_tests:
            print('filtering for min tests')
            keeps = list()
            for i, n in self._probes_df.iterrows():
                # make sure we're dealing with 1D data
                if len(n.results) >= self.min_tests:
                    keeps.append(i)
            self._probes_df = self._probes_df.iloc[keeps]
        self.posmap = {z: z for z in range(len(self._probes_df))}
        if self.limit:
            print('filtering for limit')
            keeps = range(len(self._probes_df))
            if self._sample == 'random':
                keeps = sample(keeps, self.limit)
            else:
                keeps = keeps[:self.limit]
            self._probes_df = self._probes_df.iloc[keeps]
            self.posmap = {z: keeps[z] for z in range(len(keeps))}
        self.mappos = {v: k for (k,v) in self.posmap.items()}
        self._probes_df['idx'] =  pandas.Series(range(len(self._probes_df)), index=self._probes_df.index)
        if self.prefix != 24:
            self.probes_df = self.probes_df.apply(lambda z: ChangePrefix(z, self.prefix), axis=1)

        print('done applying rules')
        gc.collect()

    @property
    def prefix(self):
        if not hasattr(self, '_prefix'):
            self._prefix = 24 # defaults to /24 answers
        return self._prefix

    @prefix.setter
    def prefix(self, val):
        self._prefix = int(val)
        self.rules_applied = False

    def group_by_probe(self):
        self.load_probes_df()
        print('grouping by probe')
        groups = self.probes_df.groupby('probe', as_index=False)
        print('aggregating probes')
        self.probes_df = groups.aggregate(lambda z: list(z))
        del groups
        self.probes_df = self.probes_df.apply(CollapsedNode, axis=1)

    def group_by_src_addr(self):
        self.load_probes_df()

    @property
    def group_mode(self):
        if not hasattr(self, '_group_mode'):
            self._group_mode = 'probe'
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
            self._limit = 0
        if not hasattr(self, '_sample'):
            self._sample = 'random'
        return self._limit

    @limit.setter
    def limit(self, limit):
        if not limit:
            self._limit = 0
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

    def get_mapped_node(self, node):
        node.results = {self.domi['i2dom'][k]: self.ipi['i2ip'][v] for (k, v) \
                in node.results.iteritems()}
        return node

    def get_raw_domain(self, key):
        return self.domi['i2dom'][key]

    def __getitem__(self, index):
        if type(index) is tuple:
            if index[1] == 1:
                return self._probes_df.iloc[self.mappos[index[0]]]
            else:
                return self._probes_df.iloc[self.posmap[index[0]]]
        return self._probes_df.iloc[index]

    def __len__(self):
        return len(self.probes_df)

    def to_(self, key):
        if key == 'resolvers' and key not in self._probes_df.columns.to_list():
            self.load_resolvers()
        def func(item):
            if type(item) in [list, ndarray, set]:
                out = zeros(len(item), dtype=object)
                for i, node in enumerate(item):
                    out[i] = func(node)
            else:
                out = self._probes_df.iloc[item][key]
            return out
        return func
