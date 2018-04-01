'''
    Collect the set of domains that are resolved on each website's landing page
    NOTE: not used (wound up using har files instead)
'''


from helpers import format_dirpath, mydir
import csv
import json
from collections import defaultdict

topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")

doms = list()


with open(supportdir+"top-1m.csv", 'r+') as f:
    reader = csv.reader(f)
    for line in reader:
        doms.append((int(line[0]), line[1]))

slds = [(z[0], ".".join(z[1].split('.')[-2:])) for z in doms]

reduced_doms = defaultdict(list)
for pos, sld in slds:
    reduced_doms[sld].append(doms[pos-1])

sld_set = set()
dom_sets = list()
for pos, sld in slds:
    if sld in sld_set:
        continue
    else:
        sld_set.add(sld)
        dom_sets.append(reduced_doms[sld])


with open(format_dirpath(topdir+"data/prelim_analysis/")+"slds.json", "w+") as f:
    json.dump(dom_sets, f)
