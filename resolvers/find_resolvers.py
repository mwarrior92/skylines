import json
from collections import defaultdict
import pickle

all_results = json.load(open("./RIPE-Atlas-measurement-30002.json","r"))

resolvers = defaultdict(set)

for sample in all_results:
    try:
        pid = sample['prb_id']
        for query in sample['resultset']:
            resolvers[pid].add(query['dst_addr'])
    except KeyError:
        pass
        # print(query)

pickle.dump(resolvers, open('./all_resolvers.pickle', 'wb'))

