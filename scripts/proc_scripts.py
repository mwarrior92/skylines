'''
imports raw experiment data
'''
from helpers import format_dirpath, mydir
import json
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF
import numpy
import pandas as pd
from pymongo import MongoClient
from IPy import IP, IPSet


################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
label = 'query_experiment'
datadir = format_dirpath(topdir+"data/"+label)
platform = "ripe_atlas"


################### load domain list ######################
with open(hardataf, 'r+') as f:
    hardata = json.load(f)
sites = list(hardata.keys())

# number of sites that include dom
site_sets = defaultdict(set)
for site in sites:
    for dom in hardata[site]['gets']:
        site_sets[dom].add(site)
ordered_doms = sorted(list(site_sets.keys()), key=lambda z: len(site_sets[z]), reverse=True)

################### NUMBER OF SITES COVERED

def sites_covered():
    print('sites covered...')
    covered = set()
    used = set()
    xy = list()
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


#################### NUMBER OF DISTINCT LINK DOMS ON EACH SITE

def distinct_links():
    print('distinct links')
    fig, ax = plt.subplots(1,1)
    ecdf = ECDF([len(set(hardata[site]['gets'])) for site in sites])
    x, y = list(ecdf.x), list(ecdf.y)
    with open('distinct_links.json', 'w+') as f:
        json.dump({'link_doms_per_site': x, 'CDF_of_sites': y}, f)



#################### NUMBER [DOMAINS / CLIENTS] PER IP

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
    print('get per ip')
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip

    return pd.DataFrame.from_records(coll.aggregate([per_ip], allowDiskUse=True))


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
    print('get per ip per dom')
    mclient = MongoClient()
    db = mclient.skyline
    coll = db.sam

    global per_ip_per_dom

    return pd.DataFrame.from_records(coll.aggregate([per_ip_per_dom], allowDiskUse=True))

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
    print('get per ip per dom')
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
    print('get per ip per dom')
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
    print('get per ip per dom')
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
