'''
imports raw experiment data
'''
from __future__ import print_function
from ripe.atlas.cousteau import Probe
from helpers import format_dirpath, mydir, isfile
import json
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import iqr
import numpy as np
import pandas as pd
from pymongo import MongoClient
from IPy import IP, IPSet
import pyasn
import inspect
from datetime import datetime


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


def get_prefix(ip):
    try:
        prefix = int(asndb.lookup(ip)[1].split('/')[1])
    except:
        prefix = 24
    return IP(ip).make_net(prefix).__str__()


def get_24(ip):
    return IP(ip).make_net(24).__str__()


def get_24_list(l, k=None):
    if k is None:
        return [get_24(z) for z in l if is_public(z)]
    else:
        return [get_24(z[k]) for z in l if is_public(z[k])]


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


def get_per_ip_per_dom():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip_per_dom

    return pd.DataFrame.from_records(coll.aggregate([per_ip_per_dom], allowDiskUse=True))


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


def agg_clients(c):
    tmp = list()
    for z in c:
        tmp.extend(z)
    return [z for z in tmp if 'src_addr' in z and is_public(z['src_addr'])]


def same_nth_24_prob_per_client_pair(data):
    print(inspect.stack()[0][3])
    with open('sites_per_dom.json', 'r+') as f:
        spd = json.load(f)
    with open('doms_per_site.json', 'r+') as f:
        dps = json.load(f)
    tmp = data._id.apply(lambda z: 'dst_addr' in z and is_public(z['dst_addr']))
    tmp = data.loc[tmp]
    print('binning')
    tmp['ip24'] = tmp._id.apply(lambda z: get_24(z['dst_addr']))
    tmp['domain'] = tmp._id.apply(lambda z: z['domain'])
    binned24 = tmp.groupby(['ip24', 'domain']).agg({'clients': agg_clients})

    ##########

    rcounts = defaultdict(list) # raw IP
    all_clients = set()
    for _, row in binned24.iterrows():
        print(str(row.name)+' '+str(len(row.clients))+' '+str(datetime.now()))
        clients = row.clients
        for i, a in enumerate(clients[:-1]):
            A = get_client_info(a['src_addr'], a['probe'])
            all_clients.add(A['ip'])
            for b in clients[i+1:]:
                B = get_client_info(b['src_addr'], b['probe'])
                rcounts[str(tuple(sorted([A['ip'], B['ip']])))].append(row.name[1])
        try:
            all_clients.add(B['ip'])
        except UnboundLocalError:
            pass
    with open('matches_24.json', 'w+') as f:
        json.dump(rcounts, f)
    with open('all_clients.json', 'w+') as f:
        json.dump(list(all_clients), f)
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    dpsr = dict()
    doms = set(tmp.domain.tolist())
    for s in dps:
        dpsr[s] = set(dps[s]).intersection(doms)
    for k in rcounts:
        sites = set()
        for d in rcounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(rcounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    with open('same_site_client_pairs24.json', 'w+') as f:
        json.dump(sscp, f)

'''
def same_nth_ip_prob(data):
    print(inspect.stack()[0][3])
    with open('sites_per_dom.json', 'r+') as f:
        spd = json.load(f)
    with open('doms_per_site.json', 'r+') as f:
        dps = json.load(f)
    tmp = data._id.apply(lambda z: 'dst_addr' in z and is_public(z['dst_addr']))
    tmp = data.loc[tmp]
    print('binning')
    tmp['ip24'] = tmp._id.apply(lambda z: get_24(z['dst_addr']))
    tmp['prefix'] = tmp._id.apply(lambda z: get_prefix(z['dst_addr']))
    tmp['domain'] = tmp._id.apply(lambda z: z['domain'])
    binned24 = tmp.groupby(['ip24', 'domain']).agg({'clients': agg_clients})
    binnedprefix = tmp.groupby(['prefix', 'domain']).agg({'clients': agg_clients})

    ##########

    rcounts = defaultdict(list) # raw IP
    pcounts = defaultdict(list) # prefix
    ccounts = defaultdict(list) # country
    acounts = defaultdict(list) # asn
    all_clients = set()
    all_prefixes = set()
    all_countries = set()
    all_asns = set()
    for _, row in binned24.iterrows():
        print(str(row.name)+' '+str(len(row.clients))+' '+str(datetime.now()))
        clients = row.clients
        for i, a in enumerate(clients[:-1]):
            if i % 100 == 0 and i > 1:
                print(str(i),end=',')
            A = get_client_info(a['src_addr'], a['probe'])
            all_clients.add(A['ip'])
            all_countries.add(A['country'])
            all_asns.add(A['asn'])
            all_prefixes.add(A['prefix'])
            for b in clients[i+1:]:
                B = get_client_info(b['src_addr'], b['probe'])
                rcounts[str(tuple(sorted([A['ip'], B['ip']])))].append(row.name[1])
                ccounts[str(tuple(sorted([A['country'], B['country']])))].append(row.name[1])
                pcounts[str(tuple(sorted([A['prefix'], B['prefix']])))].append(row.name[1])
                acounts[str(tuple(sorted([A['asn'], B['asn']])))].append(row.name[1])
        try:
            all_clients.add(B['ip'])
            all_countries.add(B['country'])
            all_asns.add(B['asn'])
            all_prefixes.add(B['prefix'])
        except UnboundLocalError:
            pass
    with open('matches_24.json', 'w+') as f:
        json.dump({'asn': acounts, 'country': ccounts, 'prefix': pcounts, 'ip24': rcounts}, f)
    with open('all_clients.json', 'w+') as f:
        json.dump({'all_clients': list(all_clients), 'all_countries': list(all_countries),
            'all_asns': list(all_asns), 'all_prefixes': list(all_prefixes)}, f)
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    dpsr = dict()
    doms = set(tmp.domain.tolist())
    sscp_out = dict()
    for s in dps:
        dpsr[s] = set(dps[s]).intersection(doms)
    for k in rcounts:
        sites = set()
        for d in rcounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(rcounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['ip24'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in pcounts:
        sites = set()
        for d in pcounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(pcounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['prefix'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in ccounts:
        sites = set()
        for d in ccounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(ccounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['country'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in acounts:
        sites = set()
        for d in acounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(acounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['asn'] = sscp
    with open('same_site_client_pairs24.json', 'w+') as f:
        json.dump(sscp_out, f)

    ##########

    rcounts = defaultdict(list) # raw IP
    pcounts = defaultdict(list) # prefix
    ccounts = defaultdict(list) # country
    acounts = defaultdict(list) # asn
    for _, row in binnedprefix.iterrows():
        print(row.name)
        clients = row.clients
        for i, a in enumerate(clients[:-1]):
            A = get_client_info(a['src_addr'], a['probe'])
            for b in clients[i+1:]:
                B = get_client_info(b['src_addr'], b['probe'])
                rcounts[str(tuple(sorted([A['ip'], B['ip']])))].append(row.name[1])
                ccounts[str(tuple(sorted([A['country'], B['country']])))].append(row.name[1])
                pcounts[str(tuple(sorted([A['prefix'], B['prefix']])))].append(row.name[1])
                acounts[str(tuple(sorted([A['asn'], B['asn']])))].append(row.name[1])
    with open('matches_prefix.json', 'w+') as f:
        json.dump({'asn': acounts, 'country': ccounts, 'prefix': pcounts, 'ipp': rcounts}, f)
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    dpsr = defaultdict(set)
    doms = set(tmp.domain.tolist())
    sscp_out = dict()
    for d in dps:
        if d in doms:
            dpsr[site].add(d)
    for k in rcounts:
        sites = set()
        for d in rcounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(rcounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['ip24'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in pcounts:
        sites = set()
        for d in pcounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(pcounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['prefix'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in ccounts:
        sites = set()
        for d in ccounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(ccounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['country'] = sscp
    sscp = defaultdict(list) # num doms from same site where same client pair matched
    for k in acounts:
        sites = set()
        for d in acounts[k]:
            sites.update(spd[d])
        for site in sites:
            n = float(len(dpsr[site].intersection(acounts[k])))
            d = float(len(dpsr[site]))
            sscp[site].append((n, d)) # should probably show this as a scatter plot
    sscp_out['asn'] = sscp
    with open('same_site_client_pairs_prefix.json', 'w+') as f:
        json.dump(sscp_out, f)

'''


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
            '_id': { 'src_addr': '$from' },
            'clients': {
                '$push': {
                    'domain': '$dst_name',
                    'probe': '$prb_id',
                    'local': '$src_addr',
                    'id': '$_id',
                    'timestamp': '$timestamp',
                    'dst_addr': '$dst_addr'
                    }
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


################### NUMBER IPs PER DOMAIN


################### DOMAIN OVERLAP

if __name__ == "__main__":
    sites_covered()
    distinct_links()
    get_per_ip()
