import json
from scoring import scoring_function
from helper import re_arrange, predict_used_alg
import sys

PROBLEMS = ['work-task-variation', 'product-and-shelves', 'tsptw', 'neighbours-rect', 'is', 'community-detection', 'word', 'foxgeesecorn', 'yumi-dynamic', 'harmony', 'cc', 'concert-hall-cap', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'efm', 'aircraft', 'triangular', 'atsp', 'black-hole', 'ctw', 'trains', 'FBD1', 'gt-sort', 'model4', 'accap', 'tower', 'hoist-benchmark', 'graph', 'monitor', 'group', 'TinyCVRP', 'JSP0', 'compression', 'wcsp', 'hitori', 'stripboard', 'portal', 'ihtc-2024-marte', 'peaceable']
PROBLEMS_2024 = ['accap', 'aircraft', 'ctw', 'community-detection', 'compression', 'concert-hall-cap', 'foxgeesecorn', 'graph', 'harmony', 'hoist-benchmark', 'monitor', 'neighbours-rect', 'efm', 'peaceable', 'portal', 'TinyCVRP', 'trains', 'triangular', 'word', 'yumi-dynamic']
PROBLEMS_2025 = ['work-task-variation', 'product-and-shelves', 'tsptw', 'is', 'cc', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'atsp', 'black-hole', 'FBD1', 'gt-sort', 'model4', 'tower', 'group', 'JSP0', 'wcsp', 'hitori', 'stripboard', 'ihtc-2024-marte']

def onevone(probs, performance_data, solvers):
    scores = {str(s):0. for s in solvers}
    s1 = solvers[0]
    s2 = solvers[1]
    for model, instance in probs:
        perfs = performance_data[model, instance]
        used_alg1 = predict_used_alg(s1, perfs)
        used_alg2 = predict_used_alg(s2, perfs)

        score_1, score_2 = scoring_function(perfs, used_alg1, used_alg2)
        scores[str(s1)] += score_1
        scores[str(s2)] += score_2
    return scores

def scoring(probs, performance_data):
    from tqdm import tqdm
    solvers = []
    with open("all_options.json") as f:
        opts = json.load(f)
    for opt in opts:
        solvers.append([tuple(o) for o in opt])
    scores = {}
    for i in tqdm(range(len(solvers))):
        s1 = solvers[i]
        for j in range(i):
            s2 = solvers[j]
            str_s1 = str(s1)
            str_s2 = str(s2)
            res = onevone(probs, performance_data, [s1, s2])
            if not str_s1 in scores:
                scores[str_s1] = {'w':[], 'l':[], 'e':[]}
            if not str_s2 in scores:
                scores[str_s2] = {'w':[], 'l':[], 'e':[]}
            if res[str_s1] > res[str_s2]:
                scores[str_s1]['w'].append(str_s2)
                scores[str_s2]['l'].append(str_s1)
            elif res[str_s1] < res[str_s2]:
                scores[str_s1]['l'].append(str_s2)
                scores[str_s2]['w'].append(str_s1)
            else:
                scores[str_s1]['e'].append(str_s2)
                scores[str_s2]['e'].append(str_s1)

    return scores

def main():
    argv = sys.argv
    if len(argv) < 4:
        print(f'usage: python {argv[0]} <problems-to-use> <json-execution-results-file> <json-results-save-file>')
        return
    problems_to_use = argv[1]
    results_file = argv[2]
    save_file = argv[3]

    if problems_to_use == '2024':
        test_problems = PROBLEMS_2024
    elif problems_to_use == '2025':
        test_problems = PROBLEMS_2025
    elif problems_to_use == 'all':
        test_problems = PROBLEMS
    else:
        print(f'unrecognised problem set {problems_to_use}. valid options are 2024, 2025 and all')
        return

    with open(results_file) as f:
        performance_data = json.load(f)

    probs = [(p['model'], p['name']) for p in performance_data['cp-sat']['1'] if p['model'] in test_problems]
    performance_data = re_arrange(performance_data)

    scores = scoring(probs, performance_data)

    for k, v in scores.items():
        if len(v['l']) == 0:
            print(f'best found static combination: {k}')

    with open(save_file, 'w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    main()
