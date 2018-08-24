from helpers import format_dirpath, mydir
import matplotlib.pyplot as plt
import json


def plot_num_sites_using_each_link_cdf(fname='num_sites_using_each_link_cdf.json'):
    with open(fname, 'r+') as f:
        data = json.load(f)

    num_sites = data['num_sites']
    cdf = data['cdf']

    fig, ax = plt.subplots(1,1)
    ax.plot(num_sites, cdf)
    ax.set_xlabel('# sites supported')
    ax.set_ylabel('CDF of domains')
    fig.savefig('num_sites_using_each_link_cdf.png')
    plt.close(fig)


def plot_num_doms_per_site_cdf(fname='num_doms_per_site_cdf.json'):
    with open(fname, 'r+') as f:
        data = json.load(f)

    fig, ax = plt.subplots(1,1)
    ax.plot(data['num_doms'], data['cdf_of_sites'])
    ax.set_xlabel('# domains used')
    ax.set_ylabel('CDF of sites')
    fig.savefig('num_doms_per_site_cdf.png')
    plt.close(fig)


def plot_num_sites_covered_by_top_n_doms(fname='num_sites_covered_by_top_n_doms.json'):
    with open(fname, 'r+') as f:
        data = json.load(f)

    used, covered, ratios = zip(*data)

    fig, ax = plt.subplots(1,1)
    ax.plot(used, covered)
    ax.set_xlabel('top domain set size')
    ax.set_ylabel('# sites using domain from set')
    fig.savefig('num_sites_covered_by_top_n_doms.png')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    y, e = zip(*ratios)
    ax.errorbar(used, y, e)
    ax.set_xlabel('top domain set size')
    ax.set_ylabel('# sites using domain from set')
    fig.savefig('fraction_links_covered_by_top_n_doms.png')
    plt.close(fig)


def plot_num_per_dst(fname1='num_per_prefix_cdf.json', fname2='num_per_24_cdf.json'):
    with open(fname1, 'r+') as f:
        per_prefix = json.load(f)

    with open(fname2, 'r+') as f:
        per_24 = json.load(f)

    fig, ax = plt.subplots(1,1)
    ax.plot(per_prefix['doms_per_ip'], per_prefix['CDF_of_ips_for_doms'], label='per prefix')
    ax.plot(per_24['doms_per_ip'], per_24['CDF_of_ips_for_doms'], label='per /24')
    ax.set_xlabel('# domains served from block')
    ax.set_ylabel('CDF of blocks')
    fig.savefig('num_domains_per_block.png')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    ax.plot(per_prefix['clients_per_ip'], per_prefix['CDF_of_ips_for_clients'], label='per prefix')
    ax.plot(per_24['clients_per_ip'], per_24['CDF_of_ips_for_clients'], label='per /24')
    ax.set_xlabel('# clients served by block')
    ax.set_ylabel('CDF of blocks')
    fig.savefig('num_clients_per_block.png')
    plt.close(fig)

