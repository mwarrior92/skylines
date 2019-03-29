import json

with open('pingdata.csv', 'r') as f:
    f.readline()
    for line in f:
        info, res = line.split(',"[')
        if info[0] != '3':
            continue
        _, prb, dstname, dstaddr, ts = info.split(',')
        res = res.replace('""', '"')
        data = {
                'prb': int(prb),
                'domain': dstname,
                'dst': dstaddr,
                'ts': int(ts),
                'res': [z['rtt'] for z in json.loads('['+res[:-2])]
                }
        with open('pingdata.json', 'a') as f2:
            f2.write(json.dumps(data)+'\n')
