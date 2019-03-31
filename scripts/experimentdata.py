import os
import inspect
import json
import pandas
from time import localtime, strftime
import cPickle as pickle
from collections import defaultdict
from IPy import IP

class ExperimentData(object):
    def __init__(self, **kwargs):
        self.open_files = dict()
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def timestr(self):
        return strftime('%Y%m%d%H%M', localtime())

    @property
    def timeid(self):
        if not hasattr(self, '_timeid'):
            self._timeid = self.timestr
        return self._timeid

    @property
    def basedir(self):
        if not hasattr(self, '_basedir'):
            self_file = os.path.abspath(inspect.stack()[0][1]) # source [1]
            basedir = "/".join(self_file.split("/")[:-2])+"/"
            self._basedir = basedir
        return self._basedir

    @basedir.setter
    def basedir(self, val):
        self._basedir = val

    @property
    def objectdir(self):
        if not hasattr(self, '_objectdir'):
            self._objectdir = self.basedir + 'objects/'
        if not os.path.exists(self._objectdir):
            os.makedirs(self._objectdir)
        return self._objectdir

    @objectdir.setter
    def objectdir(self, val):
        self._objectdir = val

    @property
    def datadir(self):
        if not hasattr(self, '_datadir'):
            self._datadir = self.basedir + 'data/'
        if not os.path.exists(self._datadir):
            os.makedirs(self._datadir)
        return self._datadir

    @datadir.setter
    def datadir(self, val):
        self._datadir = val

    @property
    def supportdir(self):
        if not hasattr(self, '_supportdir'):
            self._supportdir = self.basedir + 'support_files/'
        if not os.path.exists(self._supportdir):
            os.makedirs(self._supportdir)
        return self._supportdir

    @supportdir.setter
    def supportdir(self, val):
        self._supportdir = val

    @property
    def plotsdir(self):
        if not hasattr(self, '_plotsdir'):
            self._plotsdir = self.basedir + 'plots/'
        if not os.path.exists(self._plotsdir):
            os.makedirs(self._plotsdir)
        return self._plotsdir

    @plotsdir.setter
    def plotsdir(self, val):
        self._plotsdir = val

    def fmt_path(self, path):
        '''
        converts path from simplified name to full path
        example: "datadir/foo.json" -> "/home/user/research/data/foo.json"
        '''
        chunks = path.split('/')
        for i in range(len(chunks)):
            if chunks[i] and chunks[i].endswith('dir'):
                try:
                    chunks[i] = getattr(self, chunks[i])
                    if chunks[i].endswith('/'): # we add / later
                        chunks[i] = chunks[i][:-1]
                except AttributeError:
                    break
            else:
                break
        path = '/'.join(chunks)
        dirs = os.path.dirname(path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        return path

    def load_json(self, path):
        '''
        loads and returns json from file; path is automatically formatted
        '''
        path = self.fmt_path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, path, data):
        '''
        saves json to path all at once
        '''
        path = self.fmt_path(path)
        with open(path, 'w') as f:
            json.dump(data, f)

    def save_json_line(self, path, line):
        '''
        saves line to file with list of json entries
        '''
        path = self.fmt_path(path)
        with open(path, 'a') as f:
            f.write(json.dumps(line)+'\n')

    def iter_json(self, path):
        '''
        iterator for fragmented json file; goes one line at a time; path is
        automatically formatted
        '''
        path = self.fmt_path(path)
        if path in self.open_files:
            f = self.open_files[path]
        else:
            f = open(path, 'r')
            self.open_files[path] = f
        line = f.readline()
        if not line:
            yield None
            f.close()
            del self.open_files[path]
        else:
            yield json.loads(line)

    def load_pkl_df(self, path):
        path = self.fmt_path(path)
        return pandas.read_pickle(path)

    def save_self(self, timeid=None):
        if timeid is None:
            timeid = self.timeid
        fname = self.fmt_path('objectdir/'+self.__class__.__name__+ \
                '/'+self.timeid+'.json')
        data = dict()
        for k in self.__dict__:
            try:
                json.dumps(getattr(self, k))
                data[k] = getattr(self, k)
            except:
                if hasattr(getattr(self, k), 'save_self'):
                    getattr(getattr(self, k), 'save_self')(timeid)
        data['from_file'] = True
        with open(fname, 'w') as f:
            json.dump(data, f)

    def __del__(self):
        # close any open files so nothing is left hanging
        if hasattr(self, 'open_files'):
            paths = list(self.open_files.keys())
            for path in paths:
                try:
                    self.open_files[path].close()
                except (OSError, IOError):
                    pass
                del self.open_files[path]

class DataGetter(ExperimentData):
    @property
    def ans_prefix(self):
        if not hasattr(self, '_ans_prefix'):
            self._ans_prefix = 24 # /24 is default
        return self._ans_prefix

    @ans_prefix.setter
    def ans_prefix(self, val):
        self._ans_prefix = int(val)
        if hasattr(self, '_ip2int_mapping'):
            del self._ip2int_mapping
        if hasattr(self, '_int2ip_mapping'):
            del self._int2ip_mapping
        if hasattr(self, '_answer_counts'):
            del self._answer_counts
        if hasattr(self, '_answer_diversity'):
            del self._answer_diversity

    @property
    def dom2int_mapping(self):
        if not hasattr(self, '_dom2int_mapping'):
            with open(self.fmt_path('datadir/mappings/dom_mapping.json'), 'r') as f:
                self._dom2int_mapping = json.load(f)['dom2i']

        return self._dom2int_mapping

    def dom2int(self, dom):
        return self.dom2int_mapping[dom]

    @property
    def int2dom_mapping(self):
        if not hasattr(self, '_int2dom_mapping'):
            with open(self.fmt_path('datadir/mappings/dom_mapping.json'), 'r') as f:
                self._int2dom_mapping = json.load(f)['i2dom']
        return self._int2dom_mapping

    def int2dom(self, i):
        return self.int2dom_mapping[str(i)]

    @property
    def ip2int_mapping(self):
        if not hasattr(self, '_ip2int_mapping'):
            with open(self.fmt_path('datadir/mappings/ip_mapping.json'), 'r') as f:
                self._ip2int_mapping = json.load(f)['ip2i']
        return self._ip2int_mapping

    def ip2int(self, ip, prefix=None):
        if prefix is None:
            prefix = self.ans_prefix
        if prefix == 24:
            return self.ip2int_mapping[ip]
        else:
            prefix = ip.split('/')[1]
            ip = prefix + '/' + str(self.ans_prefix)
            return IP(ip, make_net=True).int()

    @property
    def int2ip_mapping(self):
        if not hasattr(self, '_int2ip_mapping'):
            with open(self.fmt_path('datadir/mappings/ip_mapping.json'), 'r') as f:
                self._int2ip_mapping = json.load(f)['i2ip']
        return self._int2ip_mapping

    def int2ip(self, i, prefix=None):
        if prefix is None:
            prefix = self.ans_prefix
        if prefix == 24:
            return self.int2ip_mapping[str(i)]
        else:
            ip = IP(i).strNormal()
            return IP(ip+'/'+str(prefix), make_net=True)

    @property
    def answer_counts(self):
        '''
        number of times each answer was seen
        '''
        if hasattr(self, '_answer_counts'):
            return self._answer_counts
        if self.ans_prefix == 24:
            with open(self.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
                self._answer_counts = pickle.load(f)
            return self._answer_counts
        try:
            with open(self.fmt_path('datadir/pkls/answer_counts'+str(self.ans_prefix)+'.pkl'), 'r') as f:
                self._answer_counts = pickle.load(f)
        except Exception as e:
            print(e)
            print(self.fmt_path('datadir/pkls/answer_counts'+str(self.ans_prefix)+'.pkl')+' does not exist; creating now...')
            with open(self.fmt_path('datadir/pkls/answer_counts.pkl'), 'r') as f:
                tmp_counts = pickle.load(f)
            ans_counts = defaultdict(int)
            for z in tmp_counts:
                ans = self.int2ip(z[1], 24)
                ans = self.ip2int(ans)
                ans_counts[(z[0], ans)] += tmp_counts[z]
            self._answer_counts = dict(ans_counts)
            with open(self.fmt_path('datadir/pkls/answer_counts'+str(self.ans_prefix)+'.pkl'), 'w') as f:
                pickle.dump(self._answer_counts, f)
        return self._answer_counts

    @property
    def test_counts(self):
        '''
        number of times domain was tested
        '''
        if hasattr(self, '_test_counts'):
            return self._test_counts
        with open(self.fmt_path('datadir/pkls/test_counts.pkl'), 'r') as f:
            self._test_counts = pickle.load(f)
        return self._test_counts

    @property
    def answer_diversity(self):
        '''
        number of answers seen by domain
        '''
        if hasattr(self, '_answer_diversity'):
            return self._answer_diversity
        try:
            with open(self.fmt_path('datadir/pkls/answer_diversity'+str(self.prefix)+'.pkl'), 'r') as f:
                self._answer_diversity = pickle.load(f)
        except Exception as e:
            print(e)
            print(self.fmt_path('datadir/pkls/answer_diversity'+str(self.prefix)+'.pkl')+' does not exist; creating now...')
            counts = defaultdict(int)
            for dom, _ in self.answer_counts:
                counts[dom] += 1
            self._answer_diversity = counts
            with open(self.fmt_path('datadir/pkls/answer_diversity'+str(self.prefix)+'.pkl'), 'w') as f:
                pickle.dump(self._answer_diversity, f)
        return self._answer_diversity

    def diversity(self, domain_int):
        if type(domain_int) is not int:
            domain_int = self.dom2int(domain_int)
        return self.answer_diversity[domain_int]

