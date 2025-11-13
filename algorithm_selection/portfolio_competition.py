import json
from helper import re_arrange
from helper import predict_used_alg
from scoring import scoring_function
import sys

PROBLEMS = ['work-task-variation', 'product-and-shelves', 'tsptw', 'neighbours-rect', 'is', 'community-detection', 'word', 'foxgeesecorn', 'yumi-dynamic', 'harmony', 'cc', 'concert-hall-cap', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'efm', 'aircraft', 'triangular', 'atsp', 'black-hole', 'ctw', 'trains', 'FBD1', 'gt-sort', 'model4', 'accap', 'tower', 'hoist-benchmark', 'graph', 'monitor', 'group', 'TinyCVRP', 'JSP0', 'compression', 'wcsp', 'hitori', 'stripboard', 'portal', 'ihtc-2024-marte', 'peaceable']
PROBLEMS_2024 = ['accap', 'aircraft', 'ctw', 'community-detection', 'compression', 'concert-hall-cap', 'foxgeesecorn', 'graph', 'harmony', 'hoist-benchmark', 'monitor', 'neighbours-rect', 'efm', 'peaceable', 'portal', 'TinyCVRP', 'trains', 'triangular', 'word', 'yumi-dynamic']
PROBLEMS_2025 = ['work-task-variation', 'product-and-shelves', 'tsptw', 'is', 'cc', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'atsp', 'black-hole', 'FBD1', 'gt-sort', 'model4', 'tower', 'group', 'JSP0', 'wcsp', 'hitori', 'stripboard', 'ihtc-2024-marte']

def scoring(probs, performance_data, solvers):

    solvers_str = [str(s) for s in solvers]
    scores = {s:0. for s in solvers_str}
    n = len(solvers)
    for model, instance in probs:
        perfs = performance_data[model, instance]
        for i in range(n):
            for j in range(i):
                p1 = solvers[i]
                p2 = solvers[j]
                used_alg1 = predict_used_alg(p1, perfs)
                used_alg2 = predict_used_alg(p2, perfs)

                score_1, score_2 = scoring_function(perfs, used_alg1, used_alg2)
                scores[solvers_str[i]] += score_1
                scores[solvers_str[j]] += score_2

    return scores

def main():
    if len(sys.argv) < 4:
        print("simulates the Minizinc competition on the given portfolios.")
        print(f"Usage: python {sys.argv[0]} <problem-list> <performance-file> <portfolio-file>")
        print("""<problem-list> tells the script which problems to use between 2024, 2025 or all
<performance-file> is the file where the performance data is saved (usually it is under data/results.json
<portfolio-file> is the file with all the portfolios.""")
        return
    problems_to_use = sys.argv[1]
    performance_file = sys.argv[2]
    portfolio_file = sys.argv[3]
    with open(performance_file) as f:
        performance_data = json.load(f)

    if problems_to_use == '2024':
        test_problems = PROBLEMS_2024
    elif problems_to_use == '2025':
        test_problems = PROBLEMS_2025
    elif problems_to_use == 'all':
        test_problems = PROBLEMS
    else:
        print(f'unrecognised problem set {problems_to_use}. valid options are 2024, 2025 and all')
        return

    probs = [(p['model'], p['name']) for p in performance_data['cp-sat']['1'] if p['model'] in test_problems]
    performance_data = re_arrange(performance_data)

    with open(portfolio_file) as f:
        solvers = [[tuple(s) for s in p] for p in json.load(f)]

    scores = scoring(probs, performance_data, solvers)
    for solver, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(solver, ':', score)

if __name__ == "__main__":
    main()
