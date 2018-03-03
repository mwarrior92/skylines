from pymongo import MongoClient
import json
from collections import defaultdict
from easiest.helpers import format_dirpath
from easiest.helpers import mydir
from copy import deepcopy
import numpy
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from IPy import IP

######## SET UP MONGO #########
mclient = MongoClient()
db = mclient.skyline
coll = db.dns_redirection_check

##### set up file i/o #########
data_dir = format_dirpath(mydir()+"../data/parse_dns_redirection_check/")
support_dir = format_dirpath(mydir()+"../support_files/")


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
#mg2 = coll.aggregate([group2], allowDiskUse=True)
#mg3 = coll.aggregate([group3], allowDiskUse=True)
doms = coll.distinct("domain")


def longest_prefix(a, b):
    return 32 - len(bin(abs(a.int()-b.int()))[2:])


######## find domains that share IPs ###############
def domain_overlap():
    mg4 = coll.aggregate([group4], allowDiskUse=True)
    dom_ans_sets = dict()
    dom_ans_counts = dict()
    x = list()
    xpd = list()  # median per domain
    for mg in mg4:
        dom_ans_sets[mg['domain']] = set.union(*[set(z) for z in mg['answers']])
        dom_ans_counts[mg['domain']] = numpy.mean([len(z) for z in mg['answers']])
        tmp = [len(z) for z in mg['answers']]
        x += tmp
        xpd.append(numpy.ceil(numpy.median(tmp)))

    with open(data_dir+"dom_ans_counts.json", "w+") as f:
        json.dump(dom_ans_counts, f)

    # plot number of answers per query
    fig, ax = plt.subplots(2, 1)
    ax[0].hist(x, bins='auto')
    ax[0].set_yscale("log")
    ecdf = ECDF(x)
    ax[1].plot(list(ecdf.x), list(ecdf.y))
    ax[1].set_xlim([0, 15])
    ax[1].set_xlabel("# DNS A answers for single query")
    fig.savefig(data_dir+'ans_count.png')
    plt.close(fig)

    # plot median # of IPs per answer for each domain
    fig, ax = plt.subplots()
    ecdf = ECDF(xpd)
    ax.plot(list(ecdf.x), list(ecdf.y))
    ax.set_xlim([0, 15])
    ax.set_xlabel("median # DNS A answers per query")
    fig.savefig(data_dir + 'ans_count_per_domain.png')
    plt.close(fig)

    # plot number of distinct answers per domain
    a = list() # number of distinct IPs
    b = list() # number of distinct /24 IPs
    for dom in doms:
        a.append(len(dom_ans_sets[dom]))
        b.append(len(set([".".join(z.split('.')[:-1]) for z in dom_ans_sets[dom]])))
    fig, ax = plt.subplots()
    acdf = ECDF(a)
    bcdf = ECDF(b)
    xa = list(acdf.x)
    ya = list(acdf.y)
    xb = list(bcdf.x)
    yb = list(bcdf.y)
    ap = ax.plot(xa, ya)
    bp = ax.plot(xb, yb)
    ax.set_xlim([0, 50])
    ax.legend((ap[0], bp[0]), ("# distinct IPs", "# distinct /24s"))
    fig.savefig(data_dir+'distinct_ips.png')
    plt.close(fig)

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

    pools = list()
    for dom in dom_pools:
        pools.append(list(dom_pools[dom]))

    with open(data_dir+"dom_pools.json", "w+") as f:
        json.dump(pools, f)


    with open(support_dir+"tlds.txt", "r+") as f:
        # only looking for those short tlds that appear at 2nd level
        tlds = [z for z in list(f) if len(z) == 2]

    pool_roots = list()
    for pool in pools:
        base_doms = set()
        for dom in pool:
            sub_doms = dom[:-1].split(".")
            base_dom = list()
            for sub in reversed(sub_doms):
                base_dom.insert(0, sub)
                if len(base_dom) > 1 and sub not in tlds:
                    break
            base_doms.add(".".join(base_dom))
        pool_roots.append({'roots': base_doms, 'pool': pool})

    with open(data_dir+"pool_roots.json", "w+") as f:
        tmp = [list(z['roots']) for z in pool_roots]
        json.dump(tmp, f)

    # plot number of doms per pool vs number of roots
    dom_count, root_count = zip(*[(len(d['pool']), len(d['roots'])) for d in pool_roots])
    fig, ax = plt.subplots()
    ind = numpy.arange(len(dom_count))
    width = .4
    dom_bars = ax.bar(ind, dom_count, width, color='r')
    root_bars = ax.bar(ind + width, root_count, width, color='y')
    ax.set_xlabel('domain groups')
    ax.set_yscale('log')
    ax.xaxis.set_ticklabels([])
    ax.legend((dom_bars[0], root_bars[0]), ("# domains", "# distinct SLDs"))
    fig.savefig(data_dir+'domains_groups_vs_slds_bar.png')
    plt.close(fig)

######## find answers change in back to back queries ########
def answer_flux():
    mg2 = coll.aggregate([group2], allowDiskUse=True)
    clients = defaultdict(lambda: defaultdict(dict))
    clients2 = defaultdict(lambda: defaultdict(set))
    prefix_lens_same = defaultdict(list)  # same iteration
    prefix_lens_diff = defaultdict(list)  # diff iteration
    for mg in mg2:
        for ans in mg['answers']:
            clients[mg['probe_id']][mg['domain']][ans['iteration']] = ans['A']
            clients2[mg['probe_id']][mg['domain']] = clients2[mg['probe_id']][mg['domain']].union(ans['A'])

    for client in clients.values():
        for dom in client:
            for i in client[dom]:
                if len(client[dom][i]) > 1:
                    longest = defaultdict(int)
                    for ind, a in enumerate(client[dom][i]):
                        a = IP(a)
                        for b in client[dom][i][ind+1:]:
                            b = IP(b)
                            longest[a] = max((longest_prefix(a, b), longest[a]))
                            longest[b] = max((longest_prefix(a, b), longest[b]))
                    prefix_lens_same[dom].append(longest.values())
            if len(client[dom]) == 2:
                longest = defaultdict(int)
                for a in client[dom][0]:
                    a = IP(a)
                    for b in client[dom][1]:
                        b = IP(b)
                        longest[a] = max((longest_prefix(a, b), longest[a]))
                        longest[b] = max((longest_prefix(a, b), longest[b]))
                    prefix_lens_diff[dom].append(longest.values())

    with open(data_dir+"longest_prefix_same_iter.json", "w+") as f:
        json.dump(dict(prefix_lens_same), f)

    with open(data_dir+"longest_prefix_diff_iter.json", "w+") as f:
        json.dump(dict(prefix_lens_diff), f)

    # compare client answers across different clients
    client_values = clients2.values()
    overlaps = list()
    overlaps24 = list()
    for dom in doms:
        tmp_overlaps = list()
        tmp_overlaps24 = list()
        for i, a in enumerate(client_values):
            if dom not in a:
                continue
            for b in client_values[i+1:]:
                if dom not in b:
                    continue
                n = len(a[dom].intersection(b[dom]))
                d = len(a[dom].union(b[dom]))
                tmp_overlaps.append(float(n)/float(d))
                a24 = set([".".join(z.split('.')[:-1]) for z in a[dom]])
                b24 = set([".".join(z.split('.')[:-1]) for z in b[dom]])
                n = len(a24.intersection(b24))
                d = len(a24.union(b24))
                tmp_overlaps24.append(float(n)/float(d))
        overlaps.append(numpy.mean(tmp_overlaps))
        overlaps24.append(numpy.mean(tmp_overlaps24))
    fig, ax = plt.subplots()
    ecdf = ECDF(overlaps)
    x = list(ecdf.x)
    y = list(ecdf.y)
    o = ax.plot(x, y)
    ecdf = ECDF(overlaps24)
    x = list(ecdf.x)
    y = list(ecdf.y)
    o24 = ax.plot(x, y)
    ax.legend((o[0], o24[0]), ("IPs", "/24s"))
    fig.savefig(data_dir+'client_overlap.png')
    plt.close(fig)


    # plot prefix matches for same iteration, same client
    longest = list()
    shortest = list()
    typical = list()
    for pls in prefix_lens_same.values():
        l, s = zip(*[(max(z), min(z)) for z in pls])
        longest.append(numpy.mean(l))
        shortest.append(numpy.mean(s))
        typical.append(numpy.mean([item for sublist in pls for item in sublist]))
    longest, shortest = zip(*sorted(zip(longest, shortest), key=lambda z: z[0]))
    fig, ax = plt.subplots()
    typical_ecdf = ECDF(typical)
    typical_x = list(typical_ecdf.x)
    typical_y = list(typical_ecdf.y)
    longest_ecdf = ECDF(longest)
    longest_x = list(longest_ecdf.x)
    longest_y = list(longest_ecdf.y)
    shortest_ecdf = ECDF(shortest)
    shortest_x = list(shortest_ecdf.x)
    shortest_y = list(shortest_ecdf.y)
    t = ax.plot(typical_x, typical_y)
    l = ax.plot(longest_x, longest_y)
    s = ax.plot(shortest_x, shortest_y)
    ax.set_xlabel('# bits')
    ax.legend((l[0], s[0], t[0]), ("longest", "shortest", "avg"))
    fig.savefig(data_dir+'longest_prefix_match_same_iter_cdf.png')
    plt.close(fig)

    # plot prefix matches for different iteration, same client
    longest = list()
    shortest = list()
    typical = list()
    for pld in prefix_lens_diff.values():
        l, s = zip(*[(max(z), min(z)) for z in pld])
        longest.append(numpy.mean(l))
        shortest.append(numpy.mean(s))
        typical.append(numpy.mean([item for sublist in pld for item in sublist]))
    typical, longest, shortest = zip(*sorted(zip(typical, longest, shortest), key=lambda z: z[0]))
    fig, ax = plt.subplots()
    typical_ecdf = ECDF(typical)
    typical_x = list(typical_ecdf.x)
    typical_y = list(typical_ecdf.y)
    longest_ecdf = ECDF(longest)
    longest_x = list(longest_ecdf.x)
    longest_y = list(longest_ecdf.y)
    shortest_ecdf = ECDF(shortest)
    shortest_x = list(shortest_ecdf.x)
    shortest_y = list(shortest_ecdf.y)
    t = ax.plot(typical_x, typical_y)
    l = ax.plot(longest_x, longest_y)
    s = ax.plot(shortest_x, shortest_y)
    ax.set_xlabel('# bits')
    ax.legend((l[0], s[0], t[0]), ("longest", "shortest", "avg"))
    fig.savefig(data_dir+'longest_prefix_match_diff_iter_cdf.png')
    plt.close(fig)



if __name__ == "__main__":
    #domain_overlap()
    answer_flux()
