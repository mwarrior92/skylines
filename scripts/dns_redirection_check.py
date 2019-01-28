from easiest.mms import collector, dispatcher, mdo
from easiest import cdo
from easiest.helpers import format_dirpath, mydir
import csv
import json
from pymongo import MongoClient
import time


mclient = MongoClient()
db = mclient.skyline
coll = db.dns_redirection_check

topdir = format_dirpath(mydir()+"../")
label = 'dns_redirection_check'
platform = "ripe_atlas"
# TODO find a rational way to choose these countries
countries = ["JP", "US", "AU", "FR", "IT", "BR", "IN", "DE", "ZA", "KE"]

# load top sites from Umbrella
alldoms = list()
with open(topdir+'/support_files/top-1m.csv', 'r+') as f:
    reader = csv.reader(f)
    for line in reader:
        alldoms.append(line[1])
        if len(alldoms) > 1999:
            break

size = 10
loops = len(alldoms) / size

# get clients
locs = list()
for c in countries:
    locs.append(cdo.TargetLocation())
    locs[-1].set_countries([c])

cgs = list()
for loc in locs:
    tmp_tcg = cdo.TargetClientGroup(loc, target_quantity=2)
    cgs.append(tmp_tcg.get_ClientGroup(platform))

cg = cdo.ClientGroup.merge(*cgs)
# run init test measurement to make sure nodes are working
doms = ["google.com"]
my_mdo = mdo.dns.DNS(label, query_domains=doms)
d = dispatcher.Dispatcher(my_mdo, platform, cg)
my_mro = d.dispatch()
my_mro.set('file_path',
        format_dirpath(topdir+"data/"+label+"/")+"testing_probes.json")
c = collector.SpinningCollector(my_mro, timeout=60*5, spin_time=120)
collector.wait_on_collectors([c])
with open(my_mro.get('file_path'), 'r+') as f:
    data = json.load(f)
client_info = dict()
for client in cg.get('clients'):
    client_info[client.get('probe_id')] = client.get('country_code')
good_probes = set()
for r in data['results']:
    if 'answers' in r:
        if 'A' in r['answers']:
            good_probes.add(r['prb_id'])
bad_probes = [z for z in cg.get('clients') if z.get('probe_id') not in good_probes]
cg.clients = [z for z in cg.get('clients') if z.get('probe_id') in good_probes]
locs = list()
for c in [z.get('country_code') for z in bad_probes]:
    locs.append(cdo.TargetLocation())
    locs[-1].set_countries([c])
cgs = list()
for loc in locs:
    tmp_tcg = cdo.TargetClientGroup(loc, target_quantity=1)
    cgs.append(tmp_tcg.get_ClientGroup(platform))
cgs.append(cg)
cg = cdo.ClientGroup.merge(*cgs)
for ind in range(loops):
    cg.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"clients_"+str(ind))

    print("ind is: " + str(ind) + "****************")
    doms = alldoms[ind*size:(ind+1)*size]

    # perform twice to check for fast churn
    for idx in ["i0", "i1"]:
        print(idx)
        # setup measurement
        my_mdo = mdo.dns.DNS(label, query_domains=doms)
        my_mdo.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"meas_"+str(ind)+idx)

        # deploy measurement
        d = dispatcher.Dispatcher(my_mdo, platform, cg)
        my_mro = d.dispatch()
        my_mro.set('file_path',
                format_dirpath(topdir+"data/"+label+"/")+"loop_"+str(ind)+idx+".json")

        # collect measurement results
        c = collector.SpinningCollector(my_mro, timeout=60*5, spin_time=120)

        #c.grabber_thread.join()
        collector.wait_on_collectors([c])

        with open(my_mro.get('file_path'), 'r+') as f:
            data = json.load(f)

        client_info = dict()
        pushed = 0
        for client in cg.get('clients'):
            client_info[client.get('probe_id')] = client.get('country_code')
        entries = list()
        good_probes = set()
        for r in data['results']:
            if 'answers' in r:
                if 'A' in r['answers']:
                    entries.append({
                        'probe_id': r['prb_id'],
                        'answers': r['answers'],
                        'country_code': client_info[r['prb_id']],
                        'domain': r['query_domain'],
                        'iteration': int(idx[1])
                        })
                    good_probes.add(r['prb_id'])

        coll.insert_many(entries)
        time.sleep(60)

    # refresh client set
    bad_probes = [z for z in cg.get('clients') if z.get('probe_id') not in good_probes]
    cg.clients = [z for z in cg.get('clients') if z.get('probe_id') in good_probes]
    locs = list()
    for c in [z.get('country_code') for z in bad_probes]:
        locs.append(cdo.TargetLocation())
        locs[-1].set_countries([c])
    cgs = list()

    for loc in locs:
        tmp_tcg = cdo.TargetClientGroup(loc, target_quantity=1)
        cgs.append(tmp_tcg.get_ClientGroup(platform))
    cgs.append(cg)
    cg = cdo.ClientGroup.merge(*cgs)

