import numpy as np
from geopy.distance import vincenty
from math import floor, log

def individuals_closeness(a, b):
        '''
        :params a and b: (dict(int)) key-> domain, value-> answer
        :return: closeness between a and b

        (a n b) / (b u a)
        '''
        # minimize iteration
        keys = set(a.keys()).intersection(set(b.keys()))
        if len(keys) > 0:
            aips, bips = zip(*[(a[z], b[z]) for z in keys])
            n = float(len(set(aips).intersection(bips)))
            d = float(len(aips))
            return (n/d, d)
        else:
            return (-1, 0)


def compare_individuals((agroup, bgroup, (i,j))):
    a, b = agroup.iloc[i], bgroup.iloc[j]
    closeness, count = individuals_closeness(a.results, b.results)
    distance = vincenty(a.coords, b.coords).km
    return closeness, distance, count


def compare_individuals2((a, b, pings)):
    closeness, count = individuals_closeness(a.results, b.results)
    if closeness < 0:
        return 0, 0, closeness, 0, 0, 0, count, 0, 0, 0, 0
    distance = vincenty(a.coords, b.coords).km
    bclose = floor(closeness / 0.05)
    bcount = floor(log(count, 10))
    if abs(distance) > 0:
        bdist = floor(log(abs(distance), 10))
    else:
        bdist = 0
    try:
        aping = pings.loc[a.probe].results
        bping = pings.loc[b.probe].results
        pdiff = abs(aping-bping)
        pmax = max([aping, bping])
        bping = floor(pdiff / 20)
    except KeyError:
        pdiff = -1
        pmax = -1
        bping = -1
    return a.probe, b.probe, closeness, distance, pdiff, pmax, count, bclose, bcount, bdist, bping


def get_individual_closeness((a, b, aid, bid, domtotal)):
    nd, d = individuals_closeness(a,b)
    # w = weight_by_doms(d, domtotal)
    return (nd, d, aid, bid)


def weight_by_doms(domcount, domtotal):
    return float(domcount) / float(domtotal)


def weight_by_client_space(clientsused, clientspacesize):
    return float(clientsused) / float(clientspacesize)

'''
def groups_closeness(a, b, domtotal):
    # minimize iteration
    if len(a) < len(b):
        aips, bips = zip(*[(a[z], b[z]) for z in a.keys()])
    else:
        aips, bips = zip(*[(a[z], b[z]) for z in b.keys()])

    alens = list()
    [alen.append(len(z)) for z in aips]
    blens = list()
    [blens.append(len(z)) for z in bips]
    meanmin = np.mean([min(alens[z], blens[z]) for z in range(len(alens))])

    n = [(float(len(set(aips[z]).intersection(bips[z])))/float(len(set(aips[z]).union(bips[z]))),
          min(len(aips),len(bips))) for z in range(len(aips))]
'''


def groups_simple_closeness(a, b):
    '''
    :params a and b: (dict(list)) each key is a domain, and each value is a list of dns answers
    :return: jaccard index, where a "match" for a given domain means the groups had at least one
    answer in common. Taking this route because I'm not sure how to account for large
    '''
    # minimize iteration
    if len(a) < len(b):
        aips, bips = zip(*[(a[z], b[z]) for z in a.keys()])
    else:
        aips, bips = zip(*[(a[z], b[z]) for z in b.keys()])

    alens = list()
    [alens.append(len(z)) for z in aips]
    blens = list()
    [blens.append(len(z)) for z in bips]
    meanmin = np.mean([min(alens[z], blens[z]) for z in range(len(alens))])
    d = float(len(aips))

    n = float(sum([(len(set(aips[z]).intersection(bips[z])) > 0) for z in range(len(aips))]))
    return (n/d, meanmin, d)


def weight_by_meanmin(meanmin, subnetsize):
    space = 2**(32-subnetsize) - 2
    return float(meanmin) / space


def groups_simple_distance(a, b):
    '''
    :params a and b: (dict(list)) each key is a domain, and each value is a list of dns answers
    :return: jaccard index, where a "match" for a given domain means the groups had at least one
    answer in common. Taking this route because I'm not sure how to account for large
    '''
    # minimize iteration
    if len(a) < len(b):
        aips, bips = zip(*[(a[z], b[z]) for z in a.keys()])
    else:
        aips, bips = zip(*[(a[z], b[z]) for z in b.keys()])

    alens = list()
    [alens.append(len(z)) for z in aips]
    blens = list()
    [blens.append(len(z)) for z in bips]
    meanmin = np.mean([min(alens[z], blens[z]) for z in range(len(alens))])
    d = float(len(aips))

    n = float(sum([(len(set(aips[z]).symmetric_difference(bips[z])) > 0) for z in range(len(aips))]))
    return (n/d, meanmin, d)

