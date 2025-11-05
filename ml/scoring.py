from math import floor

def scoring_function(perfs, used_alg1, used_alg2):
    _s = perfs['search']
    if _s == 'Satisfy':
        if perfs['performances'][used_alg1]['has_solution'] and not perfs['performances'][used_alg2]['has_solution']:
            return 1, 0
        elif not perfs['performances'][used_alg1]['has_solution'] and perfs['performances'][used_alg2]['has_solution']:
            return 0, 1
        else:
            ts1 = floor(perfs['performances'][used_alg1]['time'] / 1000)
            ts2 = floor(perfs['performances'][used_alg2]['time'] / 1000)
            if ts1 == ts2:
                return 0.5, 0.5
            else:
                return (ts2) / (ts1 + ts2), (ts1) / (ts1 + ts2)
    if _s == 'Minimise':
        obj1 = perfs['performances'][used_alg1]['obj']
        opt1 = perfs['performances'][used_alg1]['optimal'] == 'Optimal'
        obj2 = perfs['performances'][used_alg2]['obj']
        opt2 = perfs['performances'][used_alg2]['optimal'] == 'Optimal'
        if obj1 is None and obj2 is not None:
            assert perfs['performances'][used_alg2]['has_solution'] and not perfs['performances'][used_alg1]['has_solution']
            return 0, 1
        elif obj1 is not None and obj2 is None:
            assert perfs['performances'][used_alg1]['has_solution'] and not perfs['performances'][used_alg2]['has_solution']
            return 1, 0
        elif obj1 is None and obj2 is None:
            return 0, 0
        elif obj1 < obj2:
            return 1, 0
        elif obj2 < obj1:
            return 0, 1
        elif opt1 and not opt2:
            return 1, 0
        elif opt2 and not opt1:
            return 0, 1
        elif opt1 and opt2:
            ts1 = floor(perfs['performances'][used_alg1]['time'] / 1000)
            ts2 = floor(perfs['performances'][used_alg2]['time'] / 1000)
            if ts1 == ts2:
                return 0.5, 0.5
            else:
                return (ts2) / (ts1 + ts2), (ts1) / (ts1 + ts2)
        else:
            return 0.5, 0.5
    if _s == 'Maximise':
        obj1 = perfs['performances'][used_alg1]['obj']
        opt1 = perfs['performances'][used_alg1]['optimal'] == 'Optimal'
        obj2 = perfs['performances'][used_alg2]['obj']
        opt2 = perfs['performances'][used_alg2]['optimal'] == 'Optimal'
        if obj1 is None and obj2 is not None:
            assert perfs['performances'][used_alg2]['has_solution'] and not perfs['performances'][used_alg1]['has_solution']
            return 0, 1
        elif obj1 is not None and obj2 is None:
            assert perfs['performances'][used_alg1]['has_solution'] and not perfs['performances'][used_alg2]['has_solution']
            return 1, 0
        elif obj1 is None and obj2 is None:
            return 0, 0
        elif obj1 > obj2:
            return 1, 0
        elif obj2 > obj1:
            return 0, 1
        elif opt1 and not opt2:
            return 1, 0
        elif opt2 and not opt1:
            return 0, 1
        elif opt1 and opt2:
            ts1 = floor(perfs['performances'][used_alg1]['time'] / 1000)
            ts2 = floor(perfs['performances'][used_alg2]['time'] / 1000)
            if ts1 == ts2:
                return 0.5, 0.5
            else:
                return (ts2) / (ts1 + ts2), (ts1) / (ts1 + ts2)
        else:
            return 0.5, 0.5
    raise Exception(f"unknown search type {_s}")
