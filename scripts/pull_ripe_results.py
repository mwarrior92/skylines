import meas_handler
import helpers


if __name__ == '__main__':
    i = 86400 # -> seconds in a day
    a = 1530285305 # timestamp of end of first day
    b = a - i
    c = list()
    l = 0
    # slide date window forwards until the the last 10 attempted pulls are bad
    while len(c) == 0 or sum(c) > 0:
        print(l)
        results = meas_handler.Results(helpers.datadir+'sam'+str(l)+'.json', description='skyline_all_probes',
                stop_time__lte=a, start_time__gt=b)
        results.get_results()
        c.append(results.meas_ind)
        if len(c) > 10:
            c.pop(0)
        a -= i
        b = a - i
        l += 1
