
def sites_covered():
    print('sites covered...')
    covered = set()
    used = set()
    xy = list()
    for i, dom in enumerate(ordered_doms):
        if i > 200:
            break
        covered = covered.union(site_sets[dom])
        used.add(dom)
        ratios = list()
        for site in covered:
            d = float(len(hardata[site]['gets']))
            n = float(len([z for z in hardata[site]['gets'] if z in used]))
            ratios.append(n/d)
        xy.append((i, len(covered), numpy.median(ratios)))

def distinct_links():
    print('distinct links')
    fig, ax = plt.subplots(1,1)
    ecdf = ECDF([len(set(hardata[site]['gets'])) for site in sites])
    x, y = list(ecdf.x), list(ecdf.y)
    with open('distinct_links.json', 'w+') as f:
        json.dump({'link_doms_per_site': x, 'CDF_of_sites': y}, f)
