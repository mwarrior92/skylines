import os
import inspect
import json
import pandas
from time import localtime, strftime
import cPickle as pickle

class ExperimentData(object):
    def __init__(self, **kwargs):
        self.open_files = dict()
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @property
    def timestr(self):
        return strftime('%Y%d%H%M', localtime())

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

