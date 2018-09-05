'''
imports raw experiment data
'''
from __future__ import print_function
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
import itertools
from multiprocessing import Pool, Manager
from surveyor import get_individual_closeness
import geopy.distance
from bson.objectid import ObjectId
from reformatting import *


################### SET UP GLOBALS ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'

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


def get_client_24_groups(data=None, fname='mapped_24.pkl', fname0='mapped_clients.pkl'):
    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = make_result_to_num_mapping()
    data = data.groupby('ip24', as_index=False).agg({'results': merge_dicts,
                                                    'src_addr': lambda z: list(z),
                                                    'probe': lambda z: set(z),
                                                    'country': lambda z: list(z)[0],
                                                    'asn': lambda z: set(z),
                                                    'prefix': lambda z: set(z)})
    # filter for only groups that have more than one client
    data = data.loc[data.probe.apply(lambda z: len(z) > 1)]
    return data


def get_client_asn_groups(data=None, fname='mapped_asns.pkl', fname0='mapped_clients.pkl'):
    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = make_result_to_num_mapping()
    data = data.groupby('asn', as_index=False).agg({'results': merge_dicts,
                                                    'src_addr': lambda z: list(z),
                                                    'probe': lambda z: set(z)})
    # filter for only groups that have more than one client
    data = data.loc[data.probe.apply(lambda z: len(z) > 1)]
    return data


def get_client_country_groups(data=None, fname='mapped_country.pkl', fname0='mapped_clients.pkl'):
    if data is None:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = make_result_to_num_mapping()
    data = data.groupby('country', as_index=False).agg({'results': merge_dicts,
                                                    'src_addr': lambda z: list(z),
                                                    'probe': lambda z: set(z)})
    # filter for only groups that have more than one client
    data = data.loc[data.probe.apply(lambda z: len(z) > 1)]
    data.to_pickle(fname)
    return data


def get_client_prefix_groups(data=None, fname='mapped_prefix.pkl', fname0='mapped_clients.pkl'):
    if data is None:
        data = make_result_to_num_mapping()
    data = data.groupby('prefix', as_index=False).agg({'results': merge_dicts,
                                                    'src_addr': lambda z: list(z),
                                                    'probe': lambda z: set(z)})
    # filter for only groups that have more than one client
    data = data.loc[data.probe.apply(lambda z: len(z) > 1)]
    return data


class client_grabber(object):
    def __init__(self, data, k, total):
        self.data = data
        self.data.index = data[k]
        self.total = total
        self.i = itertools.combinations(range(len(data[k])), 2)

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


def get_probe_distances(data=None, make_maoping=True, procs=4, domtotal=302,
        fname='probe_distances.json', fname0='mapped_probes.pkl'):
    print(inspect.stack()[0][3])

    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.distances

    if data is None or make_mapping:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = get_client_probe_groups()
    data.index = data.probe
    pool = Pool(procs)
    cgrabber = client_grabber(data, 'probe', domtotal)
    print('iterating for raw distances...')
    count = 0
    recents = list()
    for c, d, a, b in pool.imap_unordered(get_individual_closeness, cgrabber):
        if c >= 0:
            count += 1
            atmp = data.loc[a]
            btmp = data.loc[b]
            mdat = {'probes': [atmp.probe, btmp.probe],
                    'countries': [atmp.country, btmp.country],
                    'asns': [list(atmp.asn), list(btmp.asn)],
                    'prefixes': [list(atmp.prefix), list(btmp.prefix)],
                    'ip24s': [list(atmp.ip24), list(btmp.ip24)],
                    'coords': [list(atmp.coords), list(btmp.coords)],
                    'dist': c,
                    'ndoms': d}
            with open(fname, 'a+') as f:
                f.write(json.dumps(mdat)+"\n")
            recents.append(mdat)
            if count % 1000 == 0:
                print([c, d])
                coll.insert_many(recents)
                recents = list()
    if len(recents) > 0:
        coll.insert_many(recents)


def get_group_dists():
    print(inspect.stack()[0][3])
    mclient = MongoClient()
    db = mclient.skyline
    print('loading collections')
    countries = db.country_dists
    ip24s = db.ip24_dists
    prefixes = db.prefix_dists
    asns = db.asn_dists
    nearness = db.nearness
    data = db.distances

    print('opening mapping files')
    with open('cip24_mapping.json', 'r+') as f:
        ip242i = json.load(f)['ip242i']
    with open('cprefix_mapping.json', 'r+') as f:
        prefix2i = json.load(f)['prefix2i']
    count = 0
    pushers = list()
    for entry in data.find():
        # handle countries
        if None not in entry['countries']:
            label = '_'.join(sorted(entry['countries']))
            if count % 1000 == 0:
                print(label, end=', ')
            d, n = entry['dist'], entry['ndoms']
            tmp = countries.find_one({'label': label})
            if tmp is None:
                countries.insert_one({'label': label, 'near': [d], 'ndoms': [n]})
            else:
                countries.update_one({'label': label},
                        {'$push': {'near': d}, '$push': {'ndoms': n}})
        # handle asns
        a, b = entry['asns']
        if len(a) == 1 and len(b) == 1:
            label = sortnjoin(a[0], b[0])
            if count % 1000 == 0:
                print(label, end=', ')
            d, n = entry['dist'], entry['ndoms']
            tmp = asns.find_one({'label': label})
            if tmp is None:
                asns.insert_one({'label': label, 'near': [d], 'ndoms': [n]})
            else:
                asns.update_one({'label': label},
                        {'$push': {'near': d}, '$push': {'ndoms': n}})
        # handle ip24s
        a, b = entry['ip24s']
        if len(a) == 1 and len(b) == 1:
            label = sortnjoin(ip242i[a[0]], ip242i[b[0]])
            if count % 1000 == 0:
                print(label, end=', ')
            d, n = entry['dist'], entry['ndoms']
            tmp = ip24s.find_one({'label': label})
            if tmp is None:
                ip24s.insert_one({'label': label, 'near': [d], 'ndoms': [n]})
            else:
                ip24s.update_one({'label': label},
                        {'$push': {'near': d}, '$push': {'ndoms': n}})
        # handle prefixes
        a, b = entry['prefixes']
        if len(a) == 1 and len(b) == 1:
            label = sortnjoin(prefix2i[a[0]], prefix2i[b[0]])
            if count % 1000 == 0:
                print(label, end=', ')
            d, n = entry['dist'], entry['ndoms']
            tmp = prefixes.find_one({'label': label})
            if tmp is None:
                prefixes.insert_one({'label': label, 'near': [d], 'ndoms': [n]})
            else:
                prefixes.update_one({'label': label},
                        {'$push': {'near': d}, '$push': {'ndoms': n}})

        # handle distance
        if None not in entry['coords']:
            geo = geopy.distance.vincenty(*entry['coords']).km
            if count % 1000 == 0:
                print(geo, end=', ')
            entry['geo'] = geo
            entry.pop('_id')
            entry['near'] = entry.pop('dist')
            pushers.append(entry)
        if count % 1000 == 0:
            print(entry['probes'])
            nearness.insert_many(pushers)
            count = 1
            pushers = list()
        else:
            count += 1

    if len(pushers) > 0:
        nearness.insert_many(pushers)

    return df


def get_raw_distances(data=None, make_maoping=True, procs=4, domtotal=302,
        fname='raw_distances.json', fname0='mapped_clients.pkl'):
    if data is None or make_mapping:
        try:
            data = pd.read_pickle(fname0)
        except Exception as e:
            print(e)
            data = make_result_to_num_mapping()
    pool = Pool(procs)
    cgrabber = client_grabber(data, 'src_addr', domtotal)
    print('iterating for raw distances...')
    count = 0
    for c, d, a, b in pool.imap_unordered(get_individual_closeness, cgrabber):
        if c >= 0:
            count += 1
            if count % 1000 == 0:
                print([c, d])
            with open(fname, 'a+') as f:
                f.write(json.dumps([c, d, str(a), str(b)])+"\n")


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


################### NUMBER IPs PER DOMAIN


################### DOMAIN OVERLAP

if __name__ == "__main__":
    #get_probe_distances()
    #get_clients_with_pings()
    get_group_dists()
