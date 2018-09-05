import pyasn
from helpers import format_dirpath, isfile
import json
from ripe.atlas.cousteau import Probe
from IPy import IP
from collections import defaultdict
asndb = pyasn.pyasn('asndb.dat')
fcached_probes = 'cached_probes.list'
import numpy as np

if isfile(fcached_probes):
    with open(fcached_probes, 'r+') as f:
        cached_probes = set(json.load(f))
else:
    cached_probes = set()



def is_public(i):
    try:
        ip = IP(i)
        return ip.iptype() == 'PUBLIC'
    except Exception as e:
        print(e)
        pass
    return False


def get_prefix(ip, cid=-1):
    prefix = 24
    try:
        prefix = int(asndb.lookup(ip)[1].split('/')[1])
    except Exception as e:
        print(e)
        try:
            prefix = int(probe_cache(cid)['prefix'].split('/')[1])
        except Exception as e:
            print(e)
            pass
    return IP(ip).make_net(prefix).__str__()


def probe_cache(cid, fpath='probe_cache/'):
    global cached_probes
    global fcached_probes
    fnum = cid / 1000
    fpath = format_dirpath(fpath)
    fname = fpath+'pc'+str(fnum)+'.json'
    ret = None
    scid = str(cid)
    if isfile(fname):
        with open(fname, 'r+') as f:
            pc = json.load(f)
            if scid in pc:
                ret = pc[scid]
    if ret is None:
        pc = dict()
        try:
            tmp = Probe(id=cid)
            pc[scid] = {'country': tmp.country_code,
                    'asn': tmp.asn_v4,
                    'prefix': tmp.prefix_v4,
                    'coords': tmp.geometry['coordinates']}
            ret = pc[scid]
        except Exception as e:
            print(e)
            print('failed to get '+scid)
        if scid in pc:
            with open(fname, 'w+') as f:
                json.dump(pc, f)
            with open(fcached_probes, 'w+') as f:
                json.dump(list(cached_probes), f)
            cached_probes.add(cid)
    if ret is None:
        raise Exception('failed to get client info')
    else:
        return ret

def get_client_info(ip, cid):
    '''
    NOTE: since this is being used for hashing, the output "ip" is actually the /24 prefix
    '''
    asn = None
    prefix = None
    try:
        prb = probe_cache(cid)
        country = prb['country']
        asn = prb['asn']
        ptmp = prb['prefix']
        if ptmp is not None and '/' in ptmp:
            prefix = int(ptmp.split('/')[1])
    except Exception as e:
        print(e)
        country = None
    try:
        prefix = int(asndb.lookup(ip)[1].split('/')[1])
    except Exception as e:
        print(e)
        if prefix is None:
            prefix = 24
    try:
        asn = int(asndb.lookup(ip)[0])
    except Exception as e:
        print(e)
        pass
    prefix = IP(ip).make_net(prefix).__str__()
    ip = IP(ip).make_net(24).__str__().split('/')[0]

    return {'country': country, 'prefix': prefix, 'asn': asn, 'ip': ip, 'id': cid}


def get_asn(ip, cid=-1):
    asn = None
    try:
        asn = int(asndb.lookup(ip)[0])
    except Exception as e:
        print(e)
        try:
            asn = probe_cache(cid)['asn']
        except Exception as e:
            print(e)
            pass
    return asn


def get_coords(cid):
    try:
        prb = probe_cache(cid)
        coords = prb['coords']
    except Exception as e:
        print(e)
        coords = None
    return coords


def get_country(cid):
    try:
        prb = probe_cache(cid)
        country = prb['country']
    except Exception as e:
        print(e)
        country = None
    return country


def get_24(ip):
    return IP(ip).make_net(24).__str__()


def get_24_list(l, k=None):
    if k is None:
        return [get_24(z) for z in l if is_public(z)]
    else:
        return [get_24(z[k]) for z in l if is_public(z[k])]


def agg_clients_raw(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return set([z['src_addr'] for z in tmp if 'src_addr' in z and \
        is_public(z['src_addr'])])


def agg_clients_24(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return set([get_24(z['src_addr']) for z in tmp if 'src_addr' in z and is_public(z['src_addr'])])


def agg_clients_prefix(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return set([get_prefix(z['src_addr'], z['probe']) for z in tmp if 'src_addr' in z and is_public(z['src_addr'])])


def agg_clients_country(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return set([get_country(z['probe']) for z in tmp if 'src_addr' in z and is_public(z['src_addr'])])


def agg_clients_asn(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return set([get_asn(z['src_addr'], z['probe']) for z in tmp if 'src_addr' in z and is_public(z['src_addr'])])


def shorten_results(r, fname='singles.json', to_tuple=False):
    '''
    reduce to results that are 1) valid / public and 2) from sites with more than one
    /24 result
    '''
    with open(fname, 'r+') as f:
        singles = json.load(f)
    if to_tuple:
        return [(get_24(z['dst_addr']), z['domain']) \
                for z in r \
                if 'dst_addr' in z and 'domain' in z \
                and is_public(z['dst_addr']) and z['domain'] not in singles]
    else:
        return {z['domain']: get_24(z['dst_addr']) \
                for z in r \
                if 'dst_addr' in z and 'domain' in z \
                and is_public(z['dst_addr']) and z['domain'] not in singles}


def is_resolved(row):
    ips, doms = zip(*list(row['results'].iteritems()))
    if len(set(ips)) > 1: # if it has more than one subnet in results, seems legit
        return True
    elif ips[0] == row['ip24']: # one subnet and matches self subnet -> bad
        return False
    elif len(set(doms)) < 4: # if only tested few doms, benefit of doubt (maybe same CDN)
        return True
    else: # if it's a lot of doms, assume bogus
        return False


def merge_dicts2(ld):
    outd = dict()
    for d in ld:
        outd.update(d)
    return outd


def merge_dicts(ld):
    outd = defaultdict(list)
    for d in ld:
        for k, v in d.iteritems():
            outd[k].append(v)
    return dict(outd)


def map_the_clients((doms, idoms, ips, iips, l1, l2, r)):
    i, row = r
    tmp = dict()
    for k, v in row.results.iteritems():
        l1.acquire()
        if k not in doms:
            doms[k] = len(doms)
            idoms[len(doms)-1] = k
        l1.release()
        l2.acquire()
        if v not in ips:
            ips[v] = len(ips)
            iips[len(ips)-1] = v
        l2.release()
        tmp[doms[k]] = ips[v]
    return (i, tmp)

def simplify_pings(results):
    ret = dict()
    for z in results:
        ret[z['domain']] = [d.popitem()[1] for d in z['ping']]
    return ret


def merge_pings(results):
    ret = defaultdict(list)
    for z in results:
        for k, v in z.iteritems():
            ret[k] += v
    return dict(ret)


def ping_medians(results):
    ret = dict()
    return [ret.update({k: np.median(v)}) for (k,v) in results.iteritems()]


def pull_from_other_df(other, coll, default=None):
    def real_func(k):
        if k in other.index:
            tmp = other.loc[k][coll]
            if type(tmp) is set:
                tmp = list(tmp)
            return tmp
        else:
            return default
    return real_func

def sortnjoin(*a):
    return '_'.join([str(z) for z in sorted(a)])
