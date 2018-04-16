import numpy
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict
from helpers import format_dirpath, mydir
import json
from pymongo import MongoClient
import difflib

##################################################################
#                           LOGGING
##################################################################
import logging
import logging.config

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# create logger
logger = logging.getLogger(__name__)
logger.debug(__name__+"logger loaded")


################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
datadir = format_dirpath(topdir+"data/dom_analysis")
label = 'query_experiment'
platform = "ripe_atlas"


with open(hardataf, 'r+') as f:
    hardata = json.load(f)
sites = list(hardata.keys())

all_doms = set()
for site in sites:
    all_doms = all_doms.union(hardata[site]['gets'])
print("number of distinct domains "+str(len(all_doms)))
with open('distinct_doms.txt', 'w+') as f:
    f.write(str(len(all_doms)))

# number of sites that include dom
site_count = defaultdict(int)
for site in sites:
    for dom in hardata[site]['gets']:
        site_count[dom] += 1

counts = [site_count[z] for z in site_count]
fig, ax = plt.subplots()
ecdf = ECDF(counts)
x = list(ecdf.x)
y = list(ecdf.y)
o = ax.plot(x, y)
ax.set_xlabel('# sites served per domain')
ax.set_ylabel('CDF')
ax.set_xlim([0, 150])
fig.savefig(datadir+'site_count.png')
plt.close(fig)
site_count_data = {'x': x, 'y': y}
with open("site_count_data.json", 'w+') as f:
    json.dump(site_count_data, f)

# coverage of top n doms
ordered_doms = sorted(list(site_count.keys()), key=lambda z: site_count[z], reverse=True)
print("orderd doms"+str(len(ordered_doms)))
with open("ordered_doms.json", "w+") as f:
    json.dump(ordered_doms, f)

i = 1
mean_ratios = list()
while i < len(ordered_doms):
    ratios = list()
    subset = set(ordered_doms[:i])
    for site in sites:
        n = len(subset.intersection(hardata[site]['gets']))
        d = len(hardata[site]['gets'])
        ratios.append(float(n)/float(d))
    mean_ratios.append((i, numpy.mean(ratios)))
    i += 1

fig, ax = plt.subplots()
x, y = list(zip(*mean_ratios))
o = ax.plot(x, y)
ax.set_xlabel('# domains used')
ax.set_ylabel('mean site coverage')
fig.savefig(datadir+'site_coverage.png')
plt.close(fig)
site_coverage_data = {'x': x, 'y': y}
with open("site_coverage_data.json", 'w+') as f:
    json.dump(site_coverage_data, f)

# amount of overlap between sites
overlaps = list()
for i, a in enumerate(sites[:-1]):
    for j, b in enumerate(sites[i+1:]):
        a_doms = set(hardata[a]['gets'])
        b_doms = hardata[b]['gets']
        o = len(a_doms.intersection(b_doms))
        '''
        overlaps[a].append(o)
        overlaps[b].append(o)
        '''
        overlaps.append(o)

fig, ax = plt.subplots()
ecdf = ECDF(overlaps)
x = list(ecdf.x)
y = list(ecdf.y)
o = ax.plot(x, y)
ax.set_xlabel('# domains shaired between site pair')
ax.set_ylabel('CDF')
fig.savefig(datadir+'shared_count.png')
plt.close(fig)
shared_count_data = {'x': x, 'y': y}
with open("shared_count_data.json", 'w+') as f:
    json.dump(shared_count_data, f)

# amount of name overlap
noverlap = defaultdict(list)
for site in sites:
    for i, a in enumerate(hardata[site]['gets'][:-1]):
        for b in hardata[site]['gets'][i+1:]:
            s = difflib.SequenceMatcher(None, a=a, b=b)
            tmp = s.find_longest_match(0, len(a), 0, len(b))
            noverlap[site].append(a[tmp.a:tmp.a+tmp.size])

variety = [numpy.mean(len(noverlap[z])) for z in noverlap]
fig, ax = plt.subplots()
ecdf = ECDF(variety)
x = list(ecdf.x)
y = list(ecdf.y)
o = ax.plot(x, y)
ax.set_xlabel("mean char overlap between a site's domains")
ax.set_ylabel('CDF')
fig.savefig(datadir+'char_overlap.png')
plt.close(fig)
char_overlap_data = {'x': x, 'y': y}
with open("char_overlap_data.json", 'w+') as f:
    json.dump(char_overlap_data, f)
