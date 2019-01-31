from helpers import format_dirpath, mydir
import json
import time
import datetime
from collections import defaultdict
import meas_handler

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
with open('config.json', 'r+') as f:
    key = json.load(f)['ripeatlas_schedule_meas_key']


if __name__ == '__main__':
    probes = meas_handler.Probes()
    probes.get_probes()
    subprobes = list()
    groups = list()
    count = 0
    for p in probes.probes:
        subprobes.append(p)
        count += 1
        if count % 50 == 0:
            count = 0
            groups.append(meas_handler.Probes(probes=subprobes))
            subprobes = list()
    for j, dom in enumerate(ordered_doms):
        if j < 150:
            continue
        print(dom)
        for i, group in enumerate(groups):
            if dom in [z['kwargs']['target'] for i,z in enumerate(group.deployments) if group.response_bools[i]]:
                continue
            success = False
            tries = 0
            while tries < 10 and not success:
                tries += 1
                success = group.deploy('ping', 'skyline_all_probes', ['skyline', 'page_links', 'harfile',
                    'experiment'], target=dom, key=key)
                if success:
                    print("success!")
                    with open(datadir+str(i)+'_group.json', 'w+') as f:
                        json.dump(group.__dict__, f)
                else:
                    print(group.responses[-1])
                    print('currently on '+dom+', which is '+str(j)+'th')
                    try:
                        rstr = json.dumps(group.responses[-1])
                        if 'same target' in rstr or 'more than' in rstr:
                            print("sleeping for 11 minutes")
                            time.sleep(11*60)
                        elif 'daily' in rstr:
                            print("waiting til tomorrow...")
                            waittiltomorrow()
                    except TypeError:
                        print("can't serialize... sleeping for a minute")
                        time.sleep(60)
                    tmp = group.pop()
                    try:
                        print(tmp)
                    except:
                        pass

