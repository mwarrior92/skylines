def closeness(a, adoms, b, bdoms):
        '''
        :param a: (tuple(set, list))
        :param b: (tuple(set, list))
        :return: closeness between a and b

        (a n b) / (b u a), where each entry is weighted using a tuple (ip, weight)

        NOTE: since every value contributes to sums twice (once from a and once from
        b), the weights are effectively half-weights. This is to account for the
        fact that different domains - or different numbers of domains - may contribute
        the same IP for a and b.

        NOTE: each domain is normalized, so domains with a lot of IPs per query
        response won't skew the results
        '''
        doms = adoms.intersection(bdoms)
        aips = {z[0] for z in a if z[1] in doms}
        bips = {z[0] for z in b if z[1] in doms}
