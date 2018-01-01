from pymongo import MongoClient
import json
from collections import defaultdict
from easiest.helpers import format_dirpath
from easiest.helpers import mydir
from copy import deepcopy
import numpy

######## SET UP MONGO #########
mclient = MongoClient()
db = mclient.skyline
coll = db.dns_redirection_check

##### set up file i/o #########
data_dir = format_dirpath(mydir()+"../data/parse_dns_redirection_check/")


# compare different domains for the same client
# do results change across domains? do domains follow correllated patterns?
group1 = {
        "$group": {
            "_id": "$probe_id",
            "answers": {
                "$push":{
                    "domain": "$domain",
                    "A": "$answers.A",
                    "mongo_id": "$_id",
                    "iteration": "$iteration",
                    "CNAME": "$answers.CNAME"
                    }
                },
            "country_code": { "$first": "$country_code" },
            "probe_id": {"$first": "$probe_id"}
            }
        }

# compare different iterations for the same client and domain
# do results change over time?
group2 = {
        "$group": {
            "_id": {
                "probe_id": "$probe_id",
                "domain": "$domain"
                },
            "probe_id": {"$first": "$probe_id"},
            "country_code": { "$first": "$country_code" },
            "domain": {"$first": "$domain"},
            "answers": {
                "$push": {
                    "A": "$answers.A",
                    "mongo_id": "$_id",
                    "iteration": "$iteration",
                    "CNAME": "$answers.CNAME"
                    }
                }
            }
        }

# compare different clients same domain
# do results change over geography?
group3 = {
        "$group": {
            "_id": {
                "domain": "$domain",
                "iteration": "$iteration"
                },
            "domain": {"$first": "$domain"},
            "answers": {
                "$push": {
                    "A": "$answers.A",
                    "mongo_id": "$_id",
                    "CNAME": "$answers.CNAME",
                    "probe_id": "$probe_id",
                    "iteration": "$iteration",
                    "country_code": "$country_code"
                    }
                }
            }
        }

group4 = {
        "$group": {
            "_id": {
                "domain": "$domain"
                },
            "domain": {"$first": "$domain"},
            "answers": { "$addToSet": "$answers.A"}
            }
        }

#mg1 = coll.aggregate([group1], allowDiskUse=True)
#mg3 = coll.aggregate([group3], allowDiskUse=True)
doms = coll.distinct("domain")

######## find domains that share IPs ###############
mg4 = coll.aggregate([group4], allowDiskUse=True)
dom_ans_sets = dict()
dom_ans_counts = dict()
for mg in mg4:
    dom_ans_sets[mg['domain']] = set.union(*[set(z) for z in mg['answers']])
    dom_ans_counts[mg['domain']] = numpy.mean([len(z) for z in mg['answers']])

with open(data_dir+"dom_ans_counts.json", "w+") as f:
    json.dump(dom_ans_counts, f)

overlaps = defaultdict(list)
for i, domA in enumerate(doms):
    for domB in doms[i+1:]:
        if len(set(dom_ans_sets[domA]).intersection(dom_ans_sets[domB])) > 0:
            overlaps[domA].append(domB)
            overlaps[domB].append(domA)

overlaps = dict(overlaps)
with open(data_dir+"dom_overlap.json", "w+") as f:
    json.dump(overlaps, f)


###### group domains connected by IPs ################
### even indirectly (eg: A shares with B, B shares with C, group has A, B, C)

dom_pools = deepcopy(overlaps)
for domA in doms:
    if domA in dom_pools:
        dom_pools[domA] = set(dom_pools[domA])
        dom_pools[domA].add(domA)
        remaining_doms = dom_pools.keys()
        for domB in remaining_doms:
            if domB != domA:
                if len(dom_pools[domA].intersection(dom_pools[domB])) > 0:
                    dom_pools[domA] = dom_pools[domA].union(dom_pools[domB])
                    del dom_pools[domB] #  throw away the spare entry

for dom in dom_pools:
    dom_pools[dom] = list(dom_pools[dom])

with open(data_dir+"dom_pools.json", "w+") as f:
    json.dump(dom_pools, f)

###### find how many
mg2 = coll.aggregate([group2], allowDiskUse=True)
