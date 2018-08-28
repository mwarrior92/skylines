'''
imports raw experiment data
'''
from __future__ import print_function
from ripe.atlas.cousteau import Probe
from helpers import format_dirpath, mydir, isfile, listfiles
import json
from collections import defaultdict, Counter
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import iqr
import numpy as np
import pandas as pd
from pymongo import MongoClient
from IPy import IP, IPSet
import pyasn
import inspect
from datetime import datetime
import itertools
from multiprocessing import Pool, Manager


################### SET UP GLOBALS ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
asndb = pyasn.pyasn('asndb.dat')
fcached_probes = 'cached_probes.list'
if isfile(fcached_probes):
    with open(fcached_probes, 'r+') as f:
        cached_probes = set(json.load(f))
else:
    cached_probes = set()

################### DOMAIN DATA COLLECTION ######################

def get_sites_per_dom():
    print(inspect.stack()[0][3])
    with open(hardataf, 'r+') as f:
        hardata = json.load(f)
    sites = list(hardata.keys())

    site_sets = defaultdict(set)
    for site in sites:
        for dom in hardata[site]['gets']:
            site_sets[dom].add(site)
    for d in site_sets:
        site_sets[d] = list(site_sets[d])
    with open('sites_per_dom.json', 'w+') as f:
        json.dump(site_sets, f)
    return site_sets


def num_sites_using_each_link_cdf(fname='sites_per_dom.json'):
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        site_sets = json.load(f)

    data = [len(site_sets[z]) for z in site_sets]
    ecdf = ECDF(data)
    num_sites, cdf = list(ecdf.x), list(ecdf.y)
    with open('num_sites_using_each_link_cdf.json', 'w+') as f:
        json.dump({'num_sites': num_sites, 'cdf': cdf}, f)
    return (num_sites, cdf)


def get_doms_per_site():
    print(inspect.stack()[0][3])
    global hardataf
    with open(hardataf, 'r+') as f:
        hardata = json.load(f)
    sites = list(hardata.keys())

    dom_sets = defaultdict(set)
    for site in sites:
        dom_sets[site] = set(hardata[site]['gets'])
    for s in dom_sets:
        dom_sets[s] = list(dom_sets[s])
    with open('doms_per_site.json', 'w+') as f:
        json.dump(dom_sets, f)
    return dom_sets


def num_doms_per_site_cdf(fname='doms_per_site.json'):
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        dom_sets = json.load(f)
    ecdf = ECDF([len(dom_sets[z]) for z in dom_sets])
    x, y = list(ecdf.x), list(ecdf.y)
    with open('num_doms_per_site_cdf.json', 'w+') as f:
        json.dump({'num_doms': x, 'cdf_of_sites': y}, f)


def num_sites_covered_by_top_n_doms(fname='sites_per_dom.json'):
    print(inspect.stack()[0][3])
    with open(hardataf, 'r+') as f:
        hardata = json.load(f)
    with open(fname, 'r+') as f:
        site_sets = json.load(f)
    ordered_doms = sorted(list(site_sets.keys()), key=lambda z: len(site_sets[z]), reverse=True)
    print('sites covered...')
    covered = set()
    used = set()
    used_vs_covered = list()
    for i, dom in enumerate(ordered_doms):
        if i > 200:
            break
        covered = covered.union(site_sets[dom])
        used.add(dom)
        ratios = list()
        for site in covered:
            d = float(len(hardata[site]['gets']))
            n = float(len([z for z in hardata[site]['gets'] if z in used]))
            ratios.append(n/d)
        used_vs_covered.append((len(used), len(covered), (np.median(ratios), iqr(ratios))))

    with open('num_sites_covered_by_top_n_doms.json', 'w+') as f:
        json.dump(used_vs_covered, f)

################### DOMAIN DATA ANALYSIS ######################

mclient = MongoClient()
db = mclient.skyline
data = db.sam

per_ip = {
        '$group': {
            '_id': '$dst_addr',
            'domains': {'$addToSet': '$dst_name'},
            'clients': {
                '$push': {
                    'src_addr': '$from',
                    'timestamp': '$timestamp',
                    'rcvd': '$rcvd',
                    'local': '$src_addr',
                    'id': '$_id',
                    'probe': '$prb_id'
                    }
                }
            }
        }


def get_per_ip():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip

    return pd.DataFrame.from_records(coll.aggregate([per_ip], allowDiskUse=True))


def is_public(i):
    try:
        ip = IP(i)
        return ip.iptype() == 'PUBLIC'
    except:
        pass
    return False

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


def nums_per_ip(data):
    print(inspect.stack()[0][3])
    tmp = data._id.apply(lambda z: is_public(z))
    tmp = data.loc[tmp]
    ips = tmp._id.tolist()
    dcounts = tmp.domains.apply(lambda z: len(z)).tolist()
    ccounts = tmp.clients.apply(lambda z: len(set(get_24_list(z, 'src_addr')))).tolist()
    with open('num_per_ip.json', 'w+') as f:
        json.dump({'IPs': ips, 'dom_counts': dcounts, 'client_counts': ccounts}, f)
    dipsubs = defaultdict(set)
    cipsubs = defaultdict(set)
    dipsubs24 = defaultdict(set)
    cipsubs24 = defaultdict(set)
    for _, row in tmp.iterrows():
        prefix = get_prefix(row._id)
        ip24 = get_24(row._id).split('/')[0]
        dipsubs[prefix].update(row.domains)
        dipsubs24[ip24].update(row.domains)
        clients = set(get_24_list(row.clients, 'src_addr'))
        cipsubs[prefix].update(clients)
        cipsubs24[ip24].update(clients)
    ips, dcounts, ccounts = zip(*[(p, len(dipsubs[p]), len(cipsubs[p])) for p in dipsubs])
    ips24, dcounts24, ccounts24 = zip(*[(p, len(dipsubs24[p]), len(cipsubs24[p])) for p in dipsubs24])
    with open('num_per_prefix.json', 'w+') as f:
        json.dump({'prefixes': ips, 'dom_counts': dcounts, 'client_counts': ccounts}, f)

    with open('num_per_24.json', 'w+') as f:
        json.dump({'prefixes': ips24, 'dom_counts': dcounts24, 'client_counts': ccounts24}, f)


def nums_per_ip_cdf():
    print(inspect.stack()[0][3])
    with open('num_per_ip.json', 'r+') as f:
        data = json.load(f)
    dcounts, ccounts = ECDF(data['dom_counts']), ECDF(data['client_counts'])
    dx, dy = list(dcounts.x), list(dcounts.y)
    cx, cy = list(ccounts.x), list(ccounts.y)
    with open('num_per_ip_cdf.json', 'w+') as f:
        json.dump({'doms_per_ip': dx, 'CDF_of_ips_for_doms': dy, 'clients_per_ip': cx,
            'CDF_of_ips_for_clients': cy}, f)
    with open('num_per_prefix.json', 'r+') as f:
        data = json.load(f)
    dcounts, ccounts = ECDF(data['dom_counts']), ECDF(data['client_counts'])
    dx, dy = list(dcounts.x), list(dcounts.y)
    cx, cy = list(ccounts.x), list(ccounts.y)
    with open('num_per_prefix_cdf.json', 'w+') as f:
        json.dump({'doms_per_ip': dx, 'CDF_of_ips_for_doms': dy, 'clients_per_ip': cx,
            'CDF_of_ips_for_clients': cy}, f)
    with open('num_per_24.json', 'r+') as f:
        data = json.load(f)
    dcounts, ccounts = ECDF(data['dom_counts']), ECDF(data['client_counts'])
    dx, dy = list(dcounts.x), list(dcounts.y)
    cx, cy = list(ccounts.x), list(ccounts.y)
    with open('num_per_24_cdf.json', 'w+') as f:
        json.dump({'doms_per_ip': dx, 'CDF_of_ips_for_doms': dy, 'clients_per_ip': cx,
            'CDF_of_ips_for_clients': cy}, f)



per_ip_per_dom = {
        '$group': {
            '_id': {
                'dst_addr': '$dst_addr',
                'domain': '$dst_name'
                },
            'clients': {
                '$push': {
                    'src_addr': '$from',
                    'probe': '$prb_id',
                    'id': '$_id',
                    'local': '$src_addr',
                    'timestamp': '$timestamp'
                    }
                }
            }
        }


def probability_curves():
    print(inspect.stack()[0][3])
    with open('matches_24.json', 'r+') as f:
        matches = json.load(f)['ip24']

    counts = sorted([len(set(matches[k])) for k in matches])
    i = 1
    m = max(counts)
    prev = len(counts)
    subset = counts
    vals = list()
    while i < m:
        prev = len(subset)
        subset = [z for z in subset if z > i]
        vals.append((i, float(len(subset)) / float(prev)))
        i += 1
    x, y = zip(*vals)
    with open('given_n_24_cdf.json', 'w+') as f:
        json.dump({'num_matches': x, 'fraction_with_more': y},f)


per_ip_per_client = {
        '$group': {
            '_id': {
                'dst_addr': '$dst_addr',
                'client': {
                    'src_addr': '$from',
                    }
                },
            'domains': {
                '$push': {
                    'domain': '$dst_name',
                    'local': '$src_addr',
                    'id': '$_id',
                    'timestamp': '$timestamp',
                    'probe': '$prb_id'
                    }
                }
            }
        }

def get_per_ip_per_client():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip_per_client

    return pd.DataFrame.from_records(coll.aggregate([per_ip_per_client], allowDiskUse=True))

per_domain = {
        '$group': {
            '_id': { 'domain': '$dst_name' },
            'clients': {
                '$push': {
                    'src_addr': '$from',
                    'probe': '$prb_id',
                    'local': '$src_addr',
                    'id': '$_id',
                    'timestamp': '$timestamp',
                    'dst_addr': '$dst_addr'
                    }
                }
            }
        }

def get_per_domain():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_domain

    return pd.DataFrame.from_records(coll.aggregate([per_domain], allowDiskUse=True))

per_client = {
        '$group': {
            '_id': '$from',
            'local': {
                '$push': '$src_addr',
                },
            'results': {
                '$push': {
                    'dst_addr': '$dst_addr',
                    'domain': '$dst_name'
                    }
                },
            'rcvd': {
                '$push': '$rcvd'
                },
            'entry_id': {
                '$push': '$_id'
                },
            'probe': {
                '$first': '$prb_id',
                }
            }
        }

def get_per_client():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_client

    return pd.DataFrame.from_records(coll.aggregate([per_client], allowDiskUse=True))


def shorten_results(r, fname='singles.json'):
    '''
    reduce to results that are 1) valid / public and 2) from sites with more than one
    /24 result
    '''
    with open(fname, 'r+') as f:
        singles = json.load(f)
    return [(get_24(z['dst_addr']), z['domain']) for z in r \
            if 'dst_addr' in z and 'domain' in z and \
            is_public(z['dst_addr']) and z['domain'] not in singles]


def is_resolved(row):
    ips, doms = zip(*row['results'])
    if len(set(ips)) > 1: # if it has more than one subnet in results, seems legit
        return True
    elif ips[0] == row['ip24']: # one subnet and matches self subnet -> bad
        return False
    elif len(set(doms)) < 4: # if only tested few doms, benefit of doubt (maybe same CDN)
        return True
    else: # if it's a lot of doms, assume bogus
        return False


def get_clients(data=None):
    print(inspect.stack()[0][3])
    if data is None:
        data = get_per_client()
    print('getting only clients with real IP addresses')
    tmp = data._id.apply(lambda z: is_public(z))
    tmp = data.loc[tmp]
    # get rid of things that didn't resolve
    tmp.results = tmp.results.apply(lambda z: shorten_results(z))
    # get rid of clients with no results
    tmp = tmp.loc[tmp.results.apply(lambda z: sum([len(i) for i in z]) > 0)]
    tmp = tmp.assign(pinfo=tmp.set_index(['_id', 'probe']).index.tolist())
    print('creating group labels')
    tmp = tmp.assign(ip24=tmp._id.apply(lambda z: get_24(z)))
    resolved = tmp.apply(is_resolved, axis=1)
    tmp = tmp.loc[resolved] # get rid of clients that didn't actually resolve stuff
    tmp = tmp.assign(prefix=tmp.pinfo.apply(lambda z: get_prefix(*z)))
    tmp = tmp.assign(country=tmp.pinfo.apply(lambda z: get_country(z[1])))
    tmp = tmp.assign(asn=tmp.pinfo.apply(lambda z: get_asn(*z)))
    tmp = tmp.assign(domains=tmp.results.apply(lambda z: set([i[1] for i in z])))
    tmp = tmp.rename(index=str, columns={'_id': 'src_addr'})
    return tmp

def dump_client_stuff(clients=None):
    if clients is None:
        clients = get_clients()
    data = clients.apply(lambda z: (z['domains'], z['src_addr']))
    with open('clientdata.json', 'w+') as f:
        json.dump(data, f)
    return data


def get_per_ip_per_dom():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip_per_dom

    return pd.DataFrame.from_records(coll.aggregate([per_ip_per_dom], allowDiskUse=True))


def get_ipdoms(data=None):
    print(inspect.stack()[0][3])
    if data is None:
        data = get_per_ip_per_dom()
    tmp = data._id.apply(lambda z: 'dst_addr' in z and is_public(z['dst_addr']))
    tmp = data.loc[tmp]
    print('getting labels')
    tmp = tmp.assign(ip24=tmp._id.apply(lambda z: get_24(z['dst_addr'])))
    tmp = tmp.assign(domain=tmp._id.apply(lambda z: z['domain']))
    tmp = tmp.assign(res_24=tmp._id.apply(lambda z: (get_24(z['dst_addr']), z['domain'])))
    # reduce to domains that actually resolve multiple destinations
    print('filtering non-multiplexed')
    multi = tmp.groupby('domain').ip24.nunique()
    tmp = tmp.loc[tmp.domain.apply(lambda z: multi[z] > 1)]
    singles = [z for z in list(multi.index) if multi[z] == 1]
    with open('singles.json', 'w+') as f:
        json.dump(singles, f)
    multi = sorted([(z, multi[z]) for z in list(multi.index) if multi[z] > 1], key=lambda z: z[1])
    with open('multi.json', 'w+') as f:
        json.dump(multi, f)
    tmp = tmp.assign(res_prefix=tmp._id.apply(lambda z: (get_prefix(z['dst_addr']), z['domain'])))
    # tmp = tmp.set_index(['ip24', 'domain']) # .agg({'clients': agg_clients})
    return tmp


def count_24_matches_raw_clients(clients=None, ipdoms=None):
    if clients is None:
        clients = get_clients()
    if ipdoms is None:
        ipdoms = get_ipdoms()
    ipdoms = ipdoms.groupby('res_24').agg({'clients': agg_clients_raw})
    print('iterating through client results')
    outdir = format_dirpath(mydir()+'matches')
    for i, client in clients.iterrows():
        print(client.src_addr)
        checked = set()
        matches = list()
        for res in client.results:
            if res[1] in checked:
                continue
            matches.extend(ipdoms.loc[[res]].clients.iloc[0])
            checked.add(res[1])
        matches = Counter(matches).most_common()
        with open(outdir+client.src_addr+'.json', 'w+') as f:
            json.dump(matches, f)
    # if it freaks out again I'm just going to write the file per client
    with open('raw_clients_24_counts.json', 'w+') as f:
        json.dump(matches, f)


def calc_counts_craw_d24(clients=None, ipdoms=None):
    if clients is None:
        clients = get_clients()
    if ipdoms is None:
        ipdoms = get_ipdoms()
    ipdoms = ipdoms.groupby('res_24').agg({'clients': agg_clients_raw})
    outdir = format_dirpath(mydir()+'matches2')
    for i, c in clients.iloc[:-1].iterrows():
        for j, c in clients.iloc[i+1:].iterrows():
            pass




################### NUMBER IPs PER DOMAIN


################### DOMAIN OVERLAP

if __name__ == "__main__":
    get_per_ip()
