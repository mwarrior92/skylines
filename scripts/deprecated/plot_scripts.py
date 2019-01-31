from helpers import format_dirpath, mydir
import inspect
import matplotlib.pyplot as plt
import json


def plot_num_sites_using_each_link_cdf(fname='num_sites_using_each_link_cdf.json'):
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        data = json.load(f)

    num_sites = data['num_sites']
    cdf = data['cdf']

    fig, ax = plt.subplots(1,1)
    ax.plot(num_sites, cdf, 'k')
    ax.set_xlabel('# pages supported')
    ax.set_ylabel('CDF of domains')
    fig.savefig('num_sites_using_each_link_cdf.png')
    plt.close(fig)


def plot_num_doms_per_site_cdf(fname='num_doms_per_site_cdf.json'):
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        data = json.load(f)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    ax.plot(data['num_doms'], data['cdf_of_sites'], 'k')
    ax.set_xlabel('# domains used')
    ax.set_ylabel('CDF of pages')
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 1])
    fig.savefig('num_doms_per_site_cdf.png')
    plt.close(fig)


def plot_num_sites_covered_by_top_n_doms(fname='num_sites_covered_by_top_n_doms.json'):
    print(inspect.stack()[0][3])
    with open(fname, 'r+') as f:
        data = json.load(f)

    used, covered, ratios = zip(*data)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    ax.plot(used, covered, 'k')
    ax.set_xlabel('domain set size')
    ax.set_ylabel('# pages covered')
    fig.savefig('num_sites_covered_by_top_n_doms.png')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    y, e = zip(*ratios)
    ax.plot(used, y, 'k')
    ax.set_xlabel('domain set size')
    ax.set_ylabel('avg fraction')
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 1])
    fig.savefig('fraction_links_covered_by_top_n_doms.png')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    y, e = zip(*ratios)
    ax.errorbar(used, y, e, fmt='k')
    ax.set_xlabel('domain set size')
    ax.set_ylabel('# pages using domain from set')
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 1])
    fig.savefig('fraction_links_covered_by_top_n_doms_err.png')
    plt.close(fig)


def plot_num_per_dst(fname1='num_per_prefix_cdf.json', fname2='num_per_24_cdf.json'):
    print(inspect.stack()[0][3])
    with open(fname1, 'r+') as f:
        per_prefix = json.load(f)

    with open(fname2, 'r+') as f:
        per_24 = json.load(f)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    ax.plot(per_prefix['doms_per_ip'], per_prefix['CDF_of_ips_for_doms'], 'k', label='per prefix')
    ax.plot(per_24['doms_per_ip'], per_24['CDF_of_ips_for_doms'], 'k--', label='per /24')
    ax.set_xlabel('# domains served from block')
    ax.set_ylabel('CDF of blocks')
    ax.legend()
    fig.savefig('num_domains_per_block.png')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    ax.plot(per_prefix['clients_per_ip'], per_prefix['CDF_of_ips_for_clients'], 'k', label='per prefix')
    ax.plot(per_24['clients_per_ip'], per_24['CDF_of_ips_for_clients'], 'k--', label='per /24')
    ax.set_xlabel('# clients served by block')
    ax.set_ylabel('CDF of blocks')
    ax.legend()
    fig.savefig('num_clients_per_block.png')
    plt.close(fig)


def plot_group_cdf(gname='country'):
    print(inspect.stack()[0][3])
    with open(gname+'_same_cdf.json', 'r+') as f:
        data = json.load(f)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3)
    ax.plot(data['closeness'], data['CDF of probes'], 'k', label='same '+gname)

    with open(gname+'_diff_cdf.json', 'r+') as f:
        data = json.load(f)
    ax.plot(data['closeness'], data['CDF of probes'], 'k--', label='diff. '+gname)
    ax.set_xlabel('pairwise closeness')
    ax.set_ylabel('CDF of '+gname+' pairs')
    ax.legend()
    fig.savefig(gname+'_cdf.png')
    plt.close(fig)


def plot_percentile_vs_counts(gname='country'):
    print(inspect.stack()[0][3])
    with open(gname+'_percentile_vs_counts.json', 'r+') as f:
        data = json.load(f)

    labels, heights = zip(*data['diff'])
    spots = labels
    lmin = min(labels)
    lmax = max(labels)

    labels = [z*10 for z in labels]
    labels1 = labels

    fig, ax = plt.subplots(1,1)
    ax.bar(spots, heights, -.5, align='edge',
            label='diff. '+gname)
    labels, heights = zip(*data['same'])
    spots = labels

    lmax = max(list(labels)+[lmax])+1
    lmin = min(list(labels)+[lmin])
    labels = [z*10 for z in labels]
    labels2 = labels

    ax.bar(spots, heights, .5, align='edge',
            label='same '+gname, hatch='/')
    labels3 = sorted(set(labels1+labels2))
    prev = labels3[0]

    labels = list()
    for label in labels3:
        while label - prev > 10:
            labels.append('')
            prev += 10
        labels.append(label)
        prev = label
    print(range(lmin, lmax))
    ax.set_xticks(range(lmin, lmax))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('mean # domains per comparison')
    ax.set_ylabel('99th percentile closeness')
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 30.5])
    ax.tick_params('x', labelrotation=90)
    ax.legend()
    fig.savefig(gname+'_percentile_vs_counts.png')
    plt.close(fig)


if __name__ == '__main__':
    plot_num_sites_using_each_link_cdf()
    plot_num_doms_per_site_cdf()
    plot_num_sites_covered_by_top_n_doms()
    plot_num_per_dst()
    plot_group_cdf('country')
    plot_group_cdf('asn')
    plot_group_cdf('ip24')
    plot_group_cdf('prefix')
    plot_percentile_vs_counts('country')
    plot_percentile_vs_counts('asn')
    plot_percentile_vs_counts('ip24')
    plot_percentile_vs_counts('prefix')
    plot_num_sites_using_each_link_cdf()
