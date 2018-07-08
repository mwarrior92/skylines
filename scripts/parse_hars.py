'''
    extract set of urls used by each site's landing page
'''
from helpers import format_dirpath, mydir, listfiles
import json
from haralyzer import HarParser
import traceback
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.distributions.empirical_distribution import ECDF
import csv
from collections import defaultdict

##################################################################
#                           LOGGING
##################################################################
import logging
import logging.config

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# create logger
logger = logging.getLogger(__name__)
logger.debug(__name__+"logger loaded")

##################################################################
#                           GLOBALS
##################################################################

topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardir = format_dirpath(topdir+'data/har0')
outdir = format_dirpath(topdir+'data/parse_hars/')

##################################################################
#                           CODE
##################################################################

harfiles = listfiles(hardir)

getlists = dict()
getcounts = list()
filters = defaultdict(int)
for ind, hf in enumerate(harfiles):
    try:
        doms = set()
        with open(hardir+hf, 'r+') as f:
            hp = HarParser(json.load(f))

        i = 0
        while 'text' not in hp.pages[0].get_requests[i]['response']['content']:
            i += 1
        src = hp.pages[0].get_requests[i]['response']['content']['text']
        status = hp.pages[0].get_requests[i]['response']['status']
        hostname = hp.pages[0].get_requests[i]['request']['url'].split('/')[2]
        if status != 200:  # avoid error pages
            filters['status_code'] += 1
            continue
        if hostname in getlists:  # avoid redundancy
            filters['redundant_host'] += 1
            continue
        if any([z0 for z0 in ['0','1','2','3','4','5','6','7','8','9'] if z0 in
            hostname.split('.')[0]]) and len(hostname.split('.')) > 2:
            filters['service_name'] += 1
            continue  # filter out server names

        for page in hp.pages:
            page_doms = [z['request']['url'].split('/')[2] for z in page.get_requests]
            doms = doms.union(page_doms)
        doms = list(set(doms))
        getcounts.append(len(doms))
        getlists[hostname] = { 'gets': doms,
                               'src_size': len(src),
                               'js': 'script' in src or '.js' in src,
                               'img': any([z in src for z in ['.jpg', '.png', '.bmp',
                               '.gif', '.jpeg', '.tif', '.jif', '.jfif', '.pdf',
                               '.swf']]) }
    except Exception as e:
        logger.error(str(e))
        traceback.print_exc()


doms = dict()
with open(supportdir+"top-1m.csv", 'r+') as f:
    reader = csv.reader(f)
    for line in reader:
        doms[line[1]] = int(line[0])
        if len(doms) > 9999:
            break

p2 = list()
for dom in getlists:
    if dom in doms:
        getlists[dom]['rank'] = doms[dom]
        p2.append((doms[dom], len(getlists[dom]['gets'])))

with open(outdir+'getlists.json', 'w+') as f:
    json.dump(getlists, f)

with open(outdir+'filtered.json', 'w+') as f:
    json.dump(filters, f)

print(numpy.median(getcounts))

fig, ax = plt.subplots()
ecdf = ECDF(getcounts)
x = list(ecdf.x)
y = list(ecdf.y)
o = ax.plot(x, y)
ax.set_xlabel('# distinct domains requested in page load')
ax.set_ylabel('CDF')
fig.savefig(outdir+'dom_count.png')
plt.close(fig)



fig, ax = plt.subplots()
p2 = sorted(p2, key=lambda z: z[0])
y, x = list(zip(*p2))
o = ax.scatter(x,y)
ax.set_xlabel('# distinct domains requested in page load')
ax.set_ylabel('rank')
fig.savefig(outdir+'rank_vs_domcount.png')
plt.close(fig)
