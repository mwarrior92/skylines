from __future__ import print_function
import ripe.atlas.cousteau as rac
import json
from datetime import datetime, timedelta
from pymongo import MongoClient

mclient = MongoClient()
db = mclient.skyline
coll = db.sqe2


class Probes(object):
    def __init__(self, **kwargs):
        self.probes = list()
        self.filter_info = list()
        self.response_bools = list()
        self.responses = list()
        self.deployments = list()
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def pop(self):
        return (self.response_bools.pop(), self.responses.pop(), self.deployments.pop())

    def save_json(self, fname):
        with open(fname, 'w+') as f:
            json.dump(self.__dict__)

    def get_probes(self, **kwargs):
        formatted = dict()
        for k in kwargs:
            if type(kwargs[k]) is list and '__in' not in k:
                formatted[k+'__in'] = kwargs[k]
            else:
                formatted[k] = kwargs[k]
        if 'status' not in formatted and 'status_name' not in formatted:
            formatted['status'] = 1
        self.probes = list(rac.ProbeRequest(**formatted))

    def deploy(self, mtype, description, tags, is_oneoff=True, af=4, start_time=None,
            resolve_on_probe=True, key=None, **kwargs):
        print([z['id'] for z in self.probes])
        sources = [rac.AtlasSource(
            type='probes',
            value=",".join([str(z['id']) for z in self.probes]),
            requested=len(self.probes))]
        kwargs['af'] = af
        kwargs['description'] = description
        kwargs['resolve_on_probe'] = resolve_on_probe
        if start_time is None:
            start_time = datetime.utcnow() + timedelta(0, 10)
        measurements = [getattr(rac, mtype.capitalize())(**kwargs)]
        req = rac.AtlasCreateRequest(
                tags=tags,
                start_time=start_time,
                is_oneoff=is_oneoff,
                measurements=measurements,
                sources=sources,
                key=key)
        b, r = req.create()
        self.deployments.append({'mtype':mtype,
            'tags':tags,
            'is_oneoff':is_oneoff,
            'start_time':str(start_time),
            'kwargs':kwargs})
        self.response_bools.append(b)
        self.responses.append(r)
        return self.response_bools[-1]

class Results(object):
    def __init__(self, outf, **search_terms):
        self.outf = outf
        if len(search_terms) > 0:
            self.search_terms = search_terms
        else:
            self.search_terms = None
        self.latest_meas_pull = None
        self.latest_result_pull = None
        self.meas_iter = None
        try:
            with open('inds.json', 'r+') as f:
                inds = json.load(f)
            self.meas_ind = inds['meas_ind']
            self.result_ind = inds['result_ind']
        except:
            self.meas_ind = -1
            self.result_ind = -1

    def get_meas_iter(self):
        self.meas_iter = rac.MeasurementRequest(**self.search_terms)

    def handle_meas(self):
        useless = ['avg', 'max', 'min', 'step']
        try:
            msm_id = self.latest_meas_pull['id']
            success, self.latest_result_pull = rac.AtlasResultsRequest(msm_id=msm_id).create()
            if success:
                for ind, res in enumerate(self.latest_result_pull):
                    if ind < self.result_ind:
                        continue
                    for k in useless:
                        try:
                            del res[k]
                        except KeyError:
                            print("missing key "+k)
                    with open(self.outf, 'a+') as f:
                        json.dump(res, f)
                        f.write("\n")
                coll.insert_many(self.latest_result_pull)
                with open('inds.json', 'w+') as f:
                    json.dump({'meas_ind': self.meas_ind, 'result_ind': self.result_ind}, f)
        except Exception as e:
            with open("result_pull_fails.json", 'a+') as f:
                json.dump({'meas': self.latest_meas_pull, 'e': str(e)}, f)
                f.write("\n")
            print("failed to pull", end=" ")
            print(e)

    def __iter__(self):
        if self.meas_iter is None:
            self.get_meas_iter()
        print(type(self.meas_iter))
        for ind, item in enumerate(self.meas_iter):
            print(item)
            if ind < self.meas_ind:
                continue
            self.latest_meas_pull = item
            self.handle_meas()
            self.meas_ind += 1
            yield item

    def get_results(self):
        for item in self:
            pass
