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
    plt.close(fig)
