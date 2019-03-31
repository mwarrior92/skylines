import skyclusters
from multiprocessing.sharedctypes import RawArray
import json

def make_domain_error(**kwargs):
    scb = skyclusters.SkyClusterBuilder(**kwargs)
    scb.nodes.keep_only({'results'})

    with open(scb.fmt_path('datadir/matrix/matrix.json'), 'r') as f:
        tmpmatrix = json.load(f)
        skyclusters.g_matrix = RawArray('d', len(tmpmatrix))
        for i, z in enumerate(tmpmatrix):
            skyclusters.g_matrix[i] = z
        scb.matrix = skyclusters.g_matrix
        del tmpmatrix
    skyclusters.gc.collect()
    scb.domain_error()
    scb.condense_domain_error()
    scb.plot_domain_error()

if __name__ == '__main__':
    make_domain_error(limit=200)
