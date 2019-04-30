import clusterAnalysis as CA
import skyclusters as SC
import json

def homogeneity_and_completeness():
    scb = SC.SkyClusterBuilder()
    with open(scb.fmt_path('datadir/matrix/matrix.json'), 'r') as f:
        scb.matrix = json.load(f)
    scb.reduce_matrix_to_sampled()
    ca = CA.ClusterAnalysis(scb=scb)
    clusters = ca.get_clusters(0.4)
    data = ca.get_homogeneity_and_completeness(clusters, 'asn')
    with open('test_homogeneity_and_completeness.json', 'a') as f:
        f.write(json.dumps([0.5, data])+'\n')


if __name__ == '__main__':
    homogeneity_and_completeness()
