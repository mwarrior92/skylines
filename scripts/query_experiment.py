from easiest.mms import collector, dispatcher, mdo
from easiest import cdo
from easiest.helpers import format_dirpath, mydir
import random
import json
from pymongo import MongoClient
import time
import numpy

################### SET UP DATABASE ######################
mclient = MongoClient()
db = mclient.skyline
coll = db.dns_big_tester


################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
datadir = format_dirpath(topdir+"/data/parse_dns_redirection_check")
label = 'dns_big_tester'
platform = "ripe_atlas"


################### load domain list ######################
with open(datadir+'kept.json', 'r+') as f:
    kept = json.load(f)
alldoms = kept.keys()


################### acquire clients ######################
tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation(), target_quantity=1000)
cg = tmp_tcg.get_ClientGroup(platform)


size = 10  # numpy.ceil(0.1*float(len(alldoms)))
sampdoms = random.sample(alldoms, size)
ind = 0
loop = 0
while True:
    # save client set state
    cg.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"clients_"+str(ind) \
                 +"_"+str(loop))

    print("ind is: " + str(ind) + "****************")
    doms = alldoms[ind*size:(ind+1)*size]

    # perform twice to check for fast churn
    # setup measurement
    my_mdo = mdo.dns.DNS(label, query_domains=doms)
    my_mdo.save_json(file_path=format_dirpath(topdir+"experiment_records/"+label+"/")+"meas_"+str(ind)+idx)

    # deploy measurement
    d = dispatcher.Dispatcher(my_mdo, platform, cg)
    my_mro = d.dispatch()
    my_mro.set('file_path', format_dirpath(topdir+"data/"+label+"/")+"loop_"+str(ind)+".json")

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
                    'iteration': ind
                    })
                good_probes.add(r['prb_id'])

    coll.insert_many(entries)
    time.sleep(60)

    # refresh client set (replace unresponsive probes)
    bad_probes = [z for z in cg.get('clients') if z.get('probe_id') not in good_probes]
    cg.clients = [z for z in cg.get('clients') if z.get('probe_id') in good_probes]
    overlap = bad_probes
    while len(overlap) > 1:  # keep reselecting until there's no redundancy
        tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation(), target_quantity=len(overlap))
        newprobes = tmp_tcg.get_ClientGroup(platform)
        overlap = newprobes.intersection(cg)
        newprobes.clients = [z for z in newprobes.get('clients') if z.get('probe_id') not in overlap]
        cg = cdo.ClientGroup.merge(cg, newprobes)

    ind += 1
    loop += 1
    # if we've looped enough times, pick a new set of probes
    if loop % 5 == 0 and loop > 0:
        loop = 0
        overlap = cg.get('probe_ids')
        tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation(), target_quantity=1000)

        cg = tmp_tcg.get_ClientGroup(platform)
        while len(overlap) > 1:  # keep reselecting until there's no redundancy
            tmp_tcg = cdo.TargetClientGroup(cdo.TargetLocation(), target_quantity=len(overlap))
            newprobes = tmp_tcg.get_ClientGroup(platform)
            overlap = newprobes.intersection(cg)
            newprobes.clients = [z for z in newprobes.get('clients') if z.get('probe_id') not in overlap]
            cg = cdo.ClientGroup.merge(cg, newprobes)

