import json
from helper import re_arrange
from helper import predict_used_alg
from scoring import scoring_function
from kmeans_as import Kmeans_AS, portfolios_2025, portfolios_2024
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

PROBLEMS = ['work-task-variation', 'product-and-shelves', 'tsptw', 'neighbours-rect', 'is', 'community-detection', 'word', 'foxgeesecorn', 'yumi-dynamic', 'harmony', 'cc', 'concert-hall-cap', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'efm', 'aircraft', 'triangular', 'atsp', 'black-hole', 'ctw', 'trains', 'FBD1', 'gt-sort', 'model4', 'accap', 'tower', 'hoist-benchmark', 'graph', 'monitor', 'group', 'TinyCVRP', 'JSP0', 'compression', 'wcsp', 'hitori', 'stripboard', 'portal', 'ihtc-2024-marte', 'peaceable']
PROBLEMS_2024 = ['accap', 'aircraft', 'ctw', 'community-detection', 'compression', 'concert-hall-cap', 'foxgeesecorn', 'graph', 'harmony', 'hoist-benchmark', 'monitor', 'neighbours-rect', 'efm', 'peaceable', 'portal', 'TinyCVRP', 'trains', 'triangular', 'word', 'yumi-dynamic']
PROBLEMS_2025 = ['work-task-variation', 'product-and-shelves', 'tsptw', 'is', 'cc', 'skill', 'cgt', 'mondoku-gcc-model-balance', 'atsp', 'black-hole', 'FBD1', 'gt-sort', 'model4', 'tower', 'group', 'JSP0', 'wcsp', 'hitori', 'stripboard', 'ihtc-2024-marte']

def scoring(probs, performance_data, features:pd.DataFrame, p:Kmeans_AS, portfolios:list[list[tuple[str,int]]]):
    solvers = [[('cp-sat',1)], [('cp-sat',8)], [('chuffed',1)], [('CPLEX',1)], [('gecode',1)], [('Picat',1)]]

    p_str = 'portfolio'

    scores = {s:0. for s in [p_str] + [str(s) for s in solvers]}
    all = [p] + solvers
    all_str = [p_str] + [str(s) for s in solvers]
    n = len(all)
    for model, instance in probs:
        perfs = performance_data[model, instance]
        for i in range(n):
            for j in range(i):
                p1 = all[i]
                p2 = all[j]
                try:
                    if isinstance(p1, Kmeans_AS):
                        x = features[features['name'] == instance].drop(columns=['name','problem']).values
                        assert isinstance(x, np.ndarray)
                        if len(x) == 0:
                            used_alg1 = ('cp-sat', '8')
                        else:
                            used_alg1 = predict_used_alg(portfolios[p.predict(x)[0]], perfs)
                    else:
                        used_alg1 = predict_used_alg(p1, perfs)
                    if isinstance(p2, Kmeans_AS):
                        x = features[features['name'] == instance].drop(columns=['name','problem']).values
                        assert isinstance(x, np.ndarray)
                        if len(x) == 0:
                            used_alg2 = ('cp-sat', '8')
                        else:
                            used_alg2 = predict_used_alg(portfolios[p.predict(x)[0]], perfs)
                    else:
                        used_alg2 = predict_used_alg(p2, perfs)
                except Exception as e:
                    print(model, instance)
                    raise e

                score_1, score_2 = scoring_function(perfs, used_alg1, used_alg2)
                scores[all_str[i]] += score_1
                scores[all_str[j]] += score_2

    return scores

def main():
    portfolio = []
    if len(sys.argv) < 6:
        print("simulates the Minizinc competition on the given solver (or portfolio of solvers) against all the others solvers available in a given problem set.")
        print(f"Usage: python {sys.argv[0]} <problem-list> <performance-file> k")
        print("""<problem-list> tells the script which problems to use between 2024 or 2025
<performance-file> is the file where the performance data is saved (usually it is under data/results.json)
<features-file> is the file where the features are saved (usually it is under data/features.csv)
<score-file> is the file where the scores for each algorithm for each instance are saved 
k is the number of elements to use with the kmeans as.""")
        return
    problems_to_use = sys.argv[1]
    performance_file = sys.argv[2]
    features_file = sys.argv[3]
    scores_file = sys.argv[4]
    k = int(sys.argv[5])
    with open(performance_file) as f:
        performance_data = json.load(f)

    with open(features_file) as f:
        features = pd.read_csv(features_file)

    with open(scores_file) as f:
        scores = json.load(f)
    as_model = Kmeans_AS(k)
    if problems_to_use == '2024':
        train_problems = PROBLEMS_2025
        test_problems = PROBLEMS_2024
        portfolios = portfolios_2024
    elif problems_to_use == '2025':
        train_problems = PROBLEMS_2024
        test_problems = PROBLEMS_2025
        portfolios = portfolios_2025
    else:
        print(f'unrecognised problem set {problems_to_use}. valid options are 2024, 2025 and all')
        return
    train_features = features[features['problem'].isin(train_problems)]
    test_features = features[features['problem'].isin(test_problems)]

    train_scores = []
    to_drop = []
    for i in range(len(train_features)):
        model = train_features.iloc[i]['problem']
        name = train_features.iloc[i]['name']
        idx = str((model, name))
        instance_scores = []
        for portfolio in portfolios:
            if idx in scores:
                instance_scores.append(scores[idx][str(portfolio)])
            else:
                to_drop.append(name)
        if len(instance_scores) > 0:
            train_scores.append(instance_scores)
    train_scores = np.array(train_scores)
    train_features = train_features.drop(train_features[train_features['name'].isin(to_drop)].index)
    train_features = train_features.drop(columns=['problem','name']).values
    assert isinstance(train_features, np.ndarray)
    as_model.train(train_features, train_scores)
    probs = [(p['model'], p['name']) for p in performance_data['cp-sat']['1'] if p['model'] in test_problems]
    performance_data = re_arrange(performance_data)

    scores = scoring(probs, performance_data, test_features, as_model, portfolios)
    for solver, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(solver, ':', score)

if __name__ == "__main__":
    main()
