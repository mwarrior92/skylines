'''
imports raw experiment data
'''
from __future__ import print_function
import sys
from helpers import format_dirpath, mydir, isfile, listfiles
import json
from collections import defaultdict, Counter
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import iqr
import numpy as np
import pandas as pd
from pymongo import MongoClient
import inspect
from datetime import datetime
from itertools import combinations, product, izip, repeat, combinations_with_replacement, imap
from multiprocessing import Pool, Manager
from surveyor import get_individual_closeness, compare_individuals, compare_individuals2
import geopy.distance
from bson.objectid import ObjectId
from reformatting import *
from math import floor


################### SET UP GLOBALS ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'

################### DOMAIN DATA COLLECTION ######################

def get_sites_per_dom():
    '''
    gets set of sites each domain appeared on
    '''
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
    '''
    plots CDF of # of sites each domain appeared on
    '''
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        site_sets = json.load(f)


    data = [len(site_sets[z]) for z in site_sets]
    with open('num_sites_using_each_link_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    ecdf = ECDF(data)
    num_sites, cdf = list(ecdf.x), list(ecdf.y)
    with open('num_sites_using_each_link_cdf.json', 'w+') as f:
        json.dump({'num_sites': num_sites, 'cdf': cdf}, f)
    return (num_sites, cdf)


def get_doms_per_site():
    '''
    gets set of domains that appeared on each site
    '''
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
    with open('doms_per_site_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    with open('doms_per_site.json', 'w+') as f:
        json.dump(dom_sets, f)
    return dom_sets


def num_doms_per_site_cdf(fname='doms_per_site.json'):
    '''
    makes CDF of number of domains that appear on each site
    '''
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        dom_sets = json.load(f)
    data = [len(dom_sets[z]) for z in dom_sets]
    with open('num_doms_per_site_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    ecdf = ECDF(data)
    x, y = list(ecdf.x), list(ecdf.y)
    with open('num_doms_per_site_cdf.json', 'w+') as f:
        json.dump({'num_doms': x, 'cdf_of_sites': y}, f)


def num_sites_covered_by_top_n_doms(fname='sites_per_dom.json'):
    '''
    gets how the number of sites covered increased with each additional domain
    considered
    '''
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
        if i > 500:
            break
        covered = covered.union(site_sets[dom])
        used.add(dom)
        ratios = list()
        for site in covered:
            d = float(len(hardata[site]['gets']))
            n = float(len([z for z in hardata[site]['gets'] if z in used]))
            ratios.append(n/d)
        used_vs_covered.append((len(used), len(covered), (np.mean(ratios), np.std(ratios))))

    with open('num_sites_covered_by_top_n_doms.json', 'w+') as f:
        json.dump(used_vs_covered, f)

################### DOMAIN DATA ANALYSIS ######################

'''
groups results by destination IP
- clients and domains are lumped into respective sets for each IP
'''

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


def nums_per_ip(data):
    '''
    gets number of domains associated with and number of clients directed to each IP
    '''
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
    data = dcounts
    with open('prefix_dom_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    data = ccounts
    with open('prefix_client_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    data = dcounts24
    with open('ip24_dom_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    data = ccounts24
    with open('ip24_client_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)

    with open('num_per_prefix.json', 'w+') as f:
        json.dump({'prefixes': ips, 'dom_counts': dcounts, 'client_counts': ccounts}, f)

    with open('num_per_24.json', 'w+') as f:
        json.dump({'prefixes': ips24, 'dom_counts': dcounts24, 'client_counts': ccounts24}, f)


def nums_per_ip_cdf():
    '''
    plot CDF of number of domains associated with each IP and number of
    clientes directed to each IP
    '''
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
    data = dcounts
    with open('dom_per_prefix_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
    data = ccounts
    with open('client_per_prefix_stats.json') as f:
        json.dump({'med': np.median(data, '75': np.percentile(data, 75), '95': np.percentile(data,
            95), '99': np.percentile(data, 99), 'mean': np.mean(data), 'std': np.std(data)}, f)
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

'''
group clients by destination IP and domain
'''

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
    '''
    get
    '''
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


def get_clients_with_pings():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    match1 = {'$match': {'rcvd': 3}}
    group1 = {'$group': {
        '_id': '$prb_id',
        'results': {'$push': {
            'ping': '$result',
            'domain': '$dst_name'}},
        'IPs': {'$push': '$from'}
        }}
    cmds = [match1, group1]
    try:
        print('trying to read raw ping results dataframe from file')
        df = pd.read_pickle('raw_results.pkl')
    except:
        print('pulling from mongo')
        df = pd.DataFrame.from_records(coll.aggregate(cmds, allowDiskUse=True))
        print('pickling')
        df.to_pickle('raw_results.pkl')
    print(df.iloc[0])
    print('simplifying pings')
    df.results = df.results.apply(simplify_pings)
    print(df.iloc[0])
    print('getting ping medians')
    df.ping_medians = df.results.apply(ping_medians)
    print(df.iloc[0])
    print('getting other columns')
    df = df.rename(index=str, columns={'_id': 'probe'})
    df.index = df.probe
    tmp = pd.read_pickle('mapped_probes.pkl')
    tmp.index = tmp.probe
    df = df.assign(country=df.probe.apply(pull_from_other_df(tmp, 'country')))
    df = df.assign(asn=df.probe.apply(pull_from_other_df(tmp, 'asn')))
    df = df.assign(ip24=df.probe.apply(pull_from_other_df(tmp, 'ip24')))
    df = df.assign(prefix=df.probe.apply(pull_from_other_df(tmp, 'prefix')))
    print(df.iloc[0])
    df.to_pickle('probe_pings.pkl')

    print('grouping by country')
    tmp = df.groupby(['country'], as_index=False).agg({'results': merge_pings})
    tmp.to_pickle('country_pings.pkl')
    print('grouping by asn')
    tmp = df.loc[df.asn.apply(lambda z: z is not None and len(z) == 1)]
    tmp.asn = tmp.asn.apply(lambda z: z.pop())
    tmp = tmp.groupby(['asn'], as_index=False).agg({'results': merge_pings})
    tmp.to_pickle('asn_pings.pkl')
    print('grouping by prefix')
    tmp = df.loc[df.prefix.apply(lambda z: z is not None and len(z) == 1)]
    tmp.prefix = tmp.prefix.apply(lambda z: z.pop())
    tmp = tmp.groupby(['prefix'], as_index=False).agg({'results': merge_pings})
    tmp.to_pickle('prefix_pings.pkl')
    print('grouping by ip24')
    tmp = df.loc[df.ip24.apply(lambda z: z is not None and len(z) == 1)]
    tmp.ip24 = tmp.ip24.apply(lambda z: z.pop())
    tmp = tmp.groupby(['ip24'], as_index=False).agg({'results': merge_pings})
    tmp.to_pickle('ip24_pings.pkl')



def get_per_client():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_client

    df = pd.DataFrame.from_records(coll.aggregate([per_client], allowDiskUse=True))
    df.to_pickle('get_per_client.pkl')
    return df


def get_clients(data=None, fname0='get_per_client.pkl', fname='get_clients.pkl'):
    print(inspect.stack()[0][3])
    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
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
    tmp.entry_id = tmp.entry_id.apply(lambda z: [str(i) for i in z])
    tmp = tmp.assign(prefix=tmp.pinfo.apply(lambda z: get_prefix(*z)))
    print('got prefixes')
    tmp = tmp.assign(coords=tmp.pinfo.apply(lambda z: get_coords(z[1])))
    print('got coordinates')
    tmp = tmp.assign(country=tmp.pinfo.apply(lambda z: get_country(z[1])))
    print('got countries')
    tmp = tmp.assign(asn=tmp.pinfo.apply(lambda z: get_asn(*z)))
    print('got ASNs')
    tmp = tmp.assign(domains=tmp.results.apply(lambda z: set([i[1] for i in z])))
    print('got domains')
    tmp = tmp.rename(index=str, columns={'_id': 'src_addr'})
    tmp.to_pickle(fname)
    print('got clients')
    return tmp


class grab_row(object):
    def __init__(self, i, args):
        self.args = args
        self.i = i

    def __iter__(self):
        return self

    def next(self):
        try:
            return tuple(self.args + [self.i.next()])
        except StopIteration:
            raise StopIteration


def make_result_to_num_mapping(data=None, update_clients=True, fname0='get_clients.pkl',
fname='mapped_clients.pkl', workers=4):
    '''
    change doms and answers into numbers so matching will be more efficient
    '''

    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = get_clients()
    m = Manager()
    p = Pool(workers)
    doms = m.dict()
    idoms = m.dict()
    ips = m.dict()
    iips = m.dict()
    l1 = m.Lock()
    l2 = m.Lock()
    ret = data
    rows = data.iterrows()
    args = grab_row(rows, [doms, idoms, ips, iips, l1, l2])
    for i, tmp in p.imap_unordered(map_the_clients, args):
        print(i)
        ret.at[i,'results'] = tmp

    with open('ip_mapping.json', 'w+') as f:
        json.dump({'ip2i': dict(ips), 'i2ip':dict(iips)}, f)

    with open('dom_mapping.json', 'w+') as f:
        json.dump({'dom2i': dict(doms), 'i2dom':dict(idoms)}, f)

    ret.to_pickle(fname)
    return ret


def get_client_probe_groups(data=None, fname='mapped_probes.pkl', fname0='mapped_clients.pkl'):
    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = make_result_to_num_mapping()
    data = data.groupby('probe', as_index=False).agg({'results': merge_dicts2,
                                                    'src_addr': lambda z: set(z),
                                                    'ip24': lambda z: set(z),
                                                    'country': lambda z: list(z)[0],
                                                    'asn': lambda z: set(z),
                                                    'prefix': lambda z: set(z),
                                                    'coords': lambda z: list(z)[0]})
    data.to_pickle(fname)
    return data


class client_grabber(object):
    def __init__(self, data, k, total):
        self.data = data
        self.data.index = data[k]
        self.total = total
        self.i = combinations(range(len(data[k])), 2)

    def __iter__(self):
        return self

    def next(self):
        try:
            tmp = self.i.next()
            a = self.data.iloc[tmp[0]]
            b = self.data.iloc[tmp[1]]
            return (a.results, b.results, a.name, b.name, self.total)
        except StopIteration:
            raise StopIteration


def get_group_distances(data=None, procs=4, domtotal=302, fname0='mapped_probes.pkl'):
    print(inspect.stack()[0][3])
    if isfile(fname0):
        data = pd.read_pickle(fname0)
    else:
        data = make_result_to_num_mapping()

    pool = Pool(procs)

    group_types = ['country', 'asn', 'prefix', 'ip24']
    # group_types = ['asn', 'prefix', 'ip24'] # 'country',
    for group_type in group_types:
        print(group_type)
        pings = pd.read_pickle(group_type+'_pings.pkl')
        pings.index = pings[group_type]
        # remove probes that moved across group labels
        if type(data[group_type].tolist()[0]) in [list, set, tuple]:
            tmp = data[group_type].apply(lambda z: len(z) == 1)
            tmp = data.loc[tmp]
            tmp[group_type] = tmp[group_type].apply(lambda z: next(iter(z)))
            grouped = tmp.groupby(group_type)
        else:
            grouped = data.groupby(group_type)
        for aname, bname in combinations_with_replacement(list(grouped.groups.keys()), 2):
            agroup = grouped.get_group(aname)
            bgroup = grouped.get_group(bname)
            if len(agroup) == 1 or len(bgroup) == 1:
                continue
            tmplist = list()
            i = izip(repeat(agroup), repeat(bgroup), product(range(len(agroup)), range(len(bgroup))))
            spins = 0
            for ret in pool.imap_unordered(compare_individuals, i, chunksize=100):
                if ret[0] >= 0:
                    tmplist.append(ret)
                    spins += 1
                    if spins % 1000 == 0:
                        print(spins, end=', ')
                        sys.stdout.flush()
            if spins > 999:
                print('\n', end='')
                print(group_type+': '+str(aname)+', '+str(bname)+'; '+str(datetime.now()))

            apings = np.median([z for y in pings.loc[aname].results.values() for z in y])
            bpings = np.median([z for y in pings.loc[bname].results.values() for z in y])
            pingdiff = abs(apings - pings)
            pingmax = max([apings, bpings])
            comps = len(agroup)*len(bgroup)
            if len(tmplist) == 0:
                continue
            elif len(tmplist) == 1:
                cmean, cmed = repeat(tmplist[0][0], 2)
                cstd = 0
                dmean, dmed = repeat(tmplist[0][1], 2)
                dstd = 0
                count = tmplist[0][2]
            else:
                closeness, distance, count = zip(*tmplist)
                cmean, dmean = np.mean(closeness), np.mean(distance)
                cmed, dmed = np.median(closeness), np.median(distance)
                cstd, dstd = np.std(closeness), np.std(distance)
                count = np.mean(count)
            if aname != bname:
                with open(group_type+'_comps_diff.json', 'a+') as f:
                    f.write(json.dumps([aname, bname, cmean, cmed, cstd, dmean, dmed, dstd, pingdiff,
                        pingmax, count, comps])+'\n')
            else:
                with open(group_type+'_comps_same.json', 'a+') as f:
                    f.write(json.dumps([aname, cmean, cmed, cstd, dmean, dmed, dstd, pingdiff,
                        pingmax, count, comps])+'\n')


def sub_group_cdfs(gname):
    data = list()
    with open(gname+'_comps_same.json', 'r+') as f:
        for line in f:
            data.append(json.loads(line)[2])
    ecdf = ECDF(data)
    x, y = list(ecdf.x), list(ecdf.y)
    with open(gname+'_same_cdf.json', 'w+') as f:
        json.dump({'closeness': x, 'CDF of probes': y}, f)

    data = list()
    with open(gname+'_comps_diff.json', 'r+') as f:
        for line in f:
            data.append(json.loads(line)[3])
    ecdf = ECDF(data)
    x, y = list(ecdf.x), list(ecdf.y)
    with open(gname+'_diff_cdf.json', 'w+') as f:
        json.dump({'closeness': x, 'CDF of probes': y}, f)


def percentiles_vs_counts(gname):
    data = defaultdict(list)
    with open(gname+'_comps_diff.json', 'r+') as f:
        for line in f:
            tmp = json.loads(line)
            skl = tmp[3]
            count = tmp[-2]
            k = floor(count / 10)
            if k > 28:
                k = 29
            k = int(k+1)
            data[k].append(skl)
    keys = sorted(data.keys())
    res = list()
    prev = min(keys)
    for k in keys:
        tmp = k - prev
        '''
        while tmp > 1:
            res.append((' ', 0))
            tmp -= 1
        '''
        res.append((k, np.percentile(data[k], 99)))
        prev = k

    data = defaultdict(list)
    with open(gname+'_comps_same.json', 'r+') as f:
        for line in f:
            tmp = json.loads(line)
            skl = tmp[2]
            count = tmp[-2]
            k = floor(count / 10)
            if k > 28:
                k = 29
            k = int(k+1)
            data[k].append(skl)
    keys = sorted(data.keys())
    ressame = list()
    prev = min(keys)
    for k in keys:
        tmp = k - prev
        '''
        while tmp > 1:
            res.append((' ', 0))
            tmp -= 1
        '''
        ressame.append((k, np.percentile(data[k], 99)))
        prev = k

    with open(gname+'_percentile_vs_counts.json', 'w+') as f:
        json.dump({'diff': res, 'same': ressame}, f)




def get_probe_distances(data=None, procs=4, domtotal=302, fname0='mapped_probes.pkl'):
    print(inspect.stack()[0][3])
    if isfile(fname0):
        data = pd.read_pickle(fname0)
    else:
        data = make_result_to_num_mapping()

    pings = pd.read_pickle('probe_pings.pkl')
    pings.results = pings.results.apply(lambda d: np.median([z for y in d.values() for z in y]))

    pool = Pool(procs)
    data.index = data.probe
    i = imap(lambda (x, y): (data.iloc[x], data.iloc[y], pings), combinations(range(len(data)), 2))
    count = 0
    print('starting to loop'+str(datetime.now()))
    for ret in pool.imap_unordered(compare_individuals2, i, chunksize=1000):
        if ret[2] < 0:
            continue
        if count % 2000 == 0:
            count = 1
            print(str(ret)+'; '+str(datetime.now()))
        else:
            count += 1
        with open('probe_v_probe.json', 'a+') as f:
            f.write(json.dumps(ret[:-4])+'\n')
        with open('closeness_'+str(ret[-4])+'.json', 'a+') as f:
            f.write(json.dumps(ret[:-4])+'\n')
        with open('count_'+str(ret[-3])+'.json', 'a+') as f:
            f.write(json.dumps(ret[:-4])+'\n')
        with open('dist_'+str(ret[-2])+'.json', 'a+') as f:
            f.write(json.dumps(ret[:-4])+'\n')
        with open('ping_'+str(ret[-1])+'.json', 'a+') as f:
            f.write(json.dumps(ret[:-4])+'\n')



################### NUMBER IPs PER DOMAIN


################### DOMAIN OVERLAP

if __name__ == "__main__":
    #get_probe_distances()
    #get_clients_with_pings()
    #get_group_dists()
    #get_group_distances()
    #get_probe_distances()
    #sub_group_cdfs('country')
    #sub_group_cdfs('ip24')
    #sub_group_cdfs('prefix')
    #sub_group_cdfs('asn')
    percentiles_vs_counts('country')
    percentiles_vs_counts('ip24')
    percentiles_vs_counts('prefix')
    percentiles_vs_counts('asn')
