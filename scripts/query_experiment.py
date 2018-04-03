from easiest.mms import collector, dispatcher, mdo
from easiest import cdo
from easiest.helpers import format_dirpath, mydir
import random
import json
from pymongo import MongoClient
import time
import numpy
import datetime

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
coll = db.query_experiment


################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
datadir = format_dirpath(topdir+"data/query_experiment")
label = 'query_experiment'
platform = "ripe_atlas"


################### load domain list ######################
with open(hardataf, 'r+') as f:
    hardata = json.load(f)
sites = list(hardata.keys())
random.shuffle(sites)
with open(supportdir+"site_order.txt", "w+") as f:
    json.dump(sites, f)



################### functions ######################
def waittiltomorrow():
    t = datetime.datetime.today()
    future = datetime.datetime(t.year,t.month,t.day,11,0)
    future += datetime.timedelta(days=1)
    time.sleep((future-t).seconds)
    global day
    global doms_today
    with open(datadir+'checked_doms'+str(day)+'.json', 'w+') as f:
        json.dump(list(doms_today), f)
    doms_today = set()

    global sites_today
    with open(datadir+'checked_sites'+str(day)+'.json', 'w+') as f:
        json.dump(list(sites_today), f)
    day += 1
    sites_today = list()


ind = 0
loop = 0
maxcred = 1000000.0
remaining = maxcred
meas_cost = 6.0
i = 0
day = 0
doms_today = set()
sites_today = list()
all_checked_sites = list()
########## GET CLIENTS ##########
print("getting clients...")
logger.debug("getting clients...")
tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation())  # get clients
cg = tmp_tcg.get_ClientGroup(platform)
while i < len(sites):
    ########## GET SITE ##########
    print("getting site")
    logger.debug("getting site")
    site = None
    if len(sites_today) > 0:  # check if we can intelligently avoid redundant meas.
        overlap = 0
        for s in sites[i:]:
            if s in all_checked_sites:
                continue
            recent = sites_today[-1]
            o = len(set(hardata[recent]['gets']).intersection(hardata[s]['gets']))
            if o > overlap:  # pick the site with the most overlap
                overlap = o
                site = s
                if overlap == len(hardata[recent]['gets']):  # check for max overlap
                    break
    if site is None:
        while site is None and i < len(sites):  # account for going out of order...
            site = sites[i]
            if site in all_checked_sites:
                site = None
                i += 1
    if site is None:  # check if we've reached end of list (unlikely...)
        break
    sites_today.append(site)
    all_checked_sites.append(site)
    # save client set state
    cg.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"clients_"+str(i))
    print(str(i)+": "+site)
    logger.debug(str(i)+": "+site)
    ########## GET DOMAINS ##########
    doms = [z for z in hardata[site]['gets'] if z not in doms_today][:20]  # get unchecked doms
    # figure out how many of the doms we can afford to check
    client_count = float(len(cg.clients))
    cost = numpy.floor(client_count*meas_cost*len(doms))
    logger.debug("cost:"+str(cost)+", clients: "+str(client_count)+\
            ", doms:"+str(len(doms))+", remaining: "+str(remaining))
    if cost > remaining:
        maxdoms = numpy.floor(remaining/(meas_cost*client_count))
        if maxdoms < 10:  # if too few, assume we should wait till tomorrow
            waittiltomorrow()
            remaining = maxcred
            continue
        else:
            doms = doms[:int(maxdoms)]
            cost = numpy.floor(client_count*meas_cost*len(doms))

    doms_today = doms_today.union(doms)
    remaining = numpy.floor(remaining - cost)

    if len(doms) > 0:
        print("doing measurement")
        logger.debug("doing measurement")
        ########## DO MEASUREMENT ##########
        # setup measurement
        my_mdo = mdo.ping.Ping(label, destinations=doms)
        my_mdo.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"meas_"+str(i))

        j = 0
        loopsize = int(50.0 / float(len(doms)))
        print(str(loopsize))
        cgs = list()
        mros = list()
        while j+loopsize <= len(cg.clients) :  # we go one client at a time because ripe rate limits
            cgs.append(cdo.ClientGroup(cg.clients[j:j+loopsize]))
            # deploy measurement
            d = dispatcher.Dispatcher(my_mdo, platform, cgs[-1])
            mros.append(d.dispatch())
            mros[-1].set('file_path', format_dirpath(topdir+"data/"+label+"/")\
                    +"".join(site.split('.'))+"_"+str(j)+".json")
            j += loopsize
        for k, my_mro in enumerate(mros):
            # collect measurement results
            print(str(k))
            c = collector.SpinningCollector(my_mro, timeout=1, spin_time=1)

            #c.grabber_thread.join()
            collector.wait_on_collectors([c])

            try:
                with open(my_mro.get('file_path'), 'r+') as f:
                    data = json.load(f)
            except IOError:
                continue

            client_info = dict()
            pushed = 0
            for client in cgs[k].get('clients'):
                client_info[client.get('probe_id')] = client.get('country_code')
            entries = list()
            for r in data['results']:
                r['idx'] = i
                r['day'] = day
                entries.append(r)
            if len(data['results']) > 0:
                coll.insert_many(entries)
        time.sleep(60*5)

    else:
        logger.debug("dom list len 0")
        print("dom list len 0")

    i += 1
    with open(datadir+'all_checked_sites.json', 'w+'):
        json.dump(all_checked_sites, f)

