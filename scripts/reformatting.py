import pyasn
from helpers import format_dirpath, isfile
import json
from ripe.atlas.cousteau import Probe
from IPy import IP
from collections import defaultdict
asndb = pyasn.pyasn('asndb.dat')


def is_public(i):
    try:
        ip = IP(i)
        return ip.iptype() == 'PUBLIC'
    except:
        pass
    return False


def get_prefix(ip, cid=-1):
    prefix = 24
    try:
        prefix = int(asndb.lookup(ip)[1].split('/')[1])
    except:
        try:
            prefix = probe_cache(cid).prefix_v4
        except:
            pass
    return IP(ip).make_net(prefix).__str__()


def probe_cache(cid, fpath='probe_cache/'):
    global cached_probes
    global fcached_probes
    fnum = cid / 1000
    fpath = format_dirpath(fpath)
    fname = fpath+'pc'+str(fnum)+'.json'
    ret = None
    if cid in cached_probes:
        with open(fnum, 'r+') as f:
            pc = json.load(f)
            ret = pc[cid]
    else:
        if isfile(fname):
            with open(fname, 'r+') as f:
                pc = json.load(f)
        else:
            pc = dict()
            try:
                pc[cid] = Probe(id=cid)
                ret = pc[cid]
            except:
                print('failed to get '+cid)
        if cid in pc:
            cached_probes.add(cid)
            with open(fname, 'w+') as f:
                json.dump(pc, f)
            with open(fcached_probes, 'w+') as f:
                json.dump(list(cached_probes), f)
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
        country = prb.country_code
        asn = prb.asn_v4
        ptmp = prb.prefix_v4
        if ptmp is not None and '/' in ptmp:
            prefix = int(ptmp.split('/')[1])
    except:
        country = None
    try:
        prefix = int(asndb.lookup(ip)[1].split('/')[1])
    except:
        if prefix is None:
            prefix = 24
    try:
        asn = int(asndb.lookup(ip)[0])
    except:
        pass
    prefix = IP(ip).make_net(prefix).__str__()
    ip = IP(ip).make_net(24).__str__().split('/')[0]

    return {'country': country, 'prefix': prefix, 'asn': asn, 'ip': ip, 'id': cid}


def get_asn(ip, cid=-1):
    asn = None
    try:
        asn = int(asndb.lookup(ip)[0])
    except:
        try:
            asn = probe_cache(cid).asn_v4
        except:
            pass
    return asn


def get_country(cid):
    try:
        prb = probe_cache(cid)
        country = prb.country_code
    except:
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


def merge_dicts(ld):
    outd = defaultdict(list)
    for d in ld:
        for k, v in d.iteritems():
            outd[k].append(v)
    return outd


def map_the_clients((doms, idoms, ips, iips, l1, l2, r)):
    i, row = r
    print(i)
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
