from easiest.mms import collector, dispatcher, mdo
from easiest import cdo
from easiest.helpers import format_dirpath, mydir
import random
import json
from pymongo import MongoClient
import time
import numpy
import datetime
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

################### SET UP DATABASE ######################
mclient = MongoClient()
db = mclient.skyline
coll = db.short_query_experiment


################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
label = 'short_query_experiment'
datadir = format_dirpath(topdir+"data/"+label)
platform = "ripe_atlas"


################### load domain list ######################
with open(hardataf, 'r+') as f:
    hardata = json.load(f)
sites = list(hardata.keys())

# number of sites that include dom
site_count = defaultdict(int)
for site in sites:
    for dom in hardata[site]['gets']:
        site_count[dom] += 1
ordered_doms = sorted(list(site_count.keys()), key=lambda z: site_count[z], reverse=True)
with open(supportdir+"site_order.txt", "w+") as f:
    json.dump(ordered_doms, f)

################### functions ######################
def waittiltomorrow():
    t = datetime.datetime.today()
    future = datetime.datetime(t.year,t.month,t.day,11,0)
    future += datetime.timedelta(days=1)
    time.sleep((future-t).seconds)

################### init globals ##################
tcg = cdo.TargetClientGroup(cdo.TargetLocation(), 1500)
cg = tcg.get_ClientGroup(platform, groupby='asn_v4', groupsize=3, globally_spread=2000)
day = 0
setsize = 66


def measure_dom_set(doms, i):
    global cg
    # setup measurement
    my_mdo = mdo.ping.Ping(label, destinations=doms)
    my_mdo.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"meas_"+str(i))
    # deploy measurement
    d = dispatcher.Dispatcher(my_mdo, platform, cg)
    my_mro = d.dispatch()
    my_mro.set('file_path', format_dirpath(topdir+"data/"+label+"/")+str(i)+".json")
    # collect measurement results
    c = collector.SpinningCollector(my_mro, timeout=20*60, spin_time=5*60)
    collector.wait_on_collectors([c])

    with open(my_mro.get('file_path'), 'r+') as f:
        data = json.load(f)

    logger.debug("swapping bad probes for hopefully good probes")
    # load data
    client_info = dict()
    for client in cg.get('clients'):
        client_info[client.get('probe_id')] = client.get('country_code')
    entries = list()
    good_probes = list()
    for r in data['results']:
        r['idx'] = i
        r['day'] = day
        entries.append(r)
        good_probes.append(r['prb_id'])
    # get rid of unresponsive probes
    bad_probes = [z for z in cg.clients if z.probe_id not in good_probes]
    locs = list()
    for c in [z.country_code for z in bad_probes]:
        locs.append(cdo.TargetLocation())
        locs[-1].set_countries([c])
    cgs = list()
    cg.clients = [z for z in cg.clients if z.probe_id in good_probes]

    for loc in locs:
        tmp_tcg = cdo.TargetClientGroup(loc, target_quantity=1)
        try:
            cgs.append(tmp_tcg.get_ClientGroup(platform))
        except Exception as e:
            logger.error(str(e))
    cgs.append(cg)
    cg = cdo.ClientGroup.merge(*cgs)
    missing = len(cg.clients)
    if missing > 0:
        try:
            tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation())
            tmpcg = tmp_tcg.get_ClientGroup(platform)
            tmpcg.clients = cg.difference(tmpcg)
            cg.clients = cg.clients + tmpcg.random_sample(missing)
        except Exception as e:
            logger.error(str(e))

    # push data to mongodb
    if len(data['results']) > 0:
        coll.insert_many(entries)
    time.sleep(60*5)

ind = 0
redos = list()
while ind+setsize < len(ordered_doms):
    domset = ordered_doms[ind:ind+setsize]
    print(domset)
    print(ind)
    print('measuring')
    try:
        measure_dom_set(domset, ind)
    except Exception as e:
        logger.error("MUST_REDO: "+str(ind)+"_"+str(ind+setsize)+"; "+str(e))
        redos.append(ind)
    print('waiting til tomorrow')
    waittiltomorrow()
    ind += setsize

for ind in redos:
    domset = ordered_doms[ind:ind+setsize]
    print(domset)
    print(ind)
    print('measuring')
    try:
        measure_dom_set(domset, ind)
    except Exception as e:
        logger.error("FAILED_TO_REDO: "+str(ind)+"_"+str(ind+setsize)+"; "+str(e))
    print('waiting til tomorrow')
    waittiltomorrow()
    ind += setsize

