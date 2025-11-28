import random
import numpy as np
import torch
import pandas as pd
from typing import Literal
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from classification_models.kmeans_classifier import KmeansClassifier
from classification_models.knn_classifier import KnnClassifier
from classification_models.neural_network import NN
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

ModelTypes = Literal['decisionTree','gradientBoost', 'neuralNetwork', 'supportVectorMachine', 'kmeans', 'knn', 'sgd']

def set_seed(seed=42):
    random.seed(seed)                     # Python built-in random
    np.random.seed(seed)                  # Numpy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU (if using)
    torch.cuda.manual_seed_all(seed)      # All GPUs (if using multi-GPU)

    # For deterministic behavior (may affect performance)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def get_hyperparameters(model_type:ModelTypes) -> dict:
    if model_type == 'decisionTree':
        return {
                "criterion": ["gini", "entropy"],             # Splitting criteria
                "splitter": ["best", "random"],              # Split strategy
                "max_depth": [None, 5, 10, 15, 20, 30],      # Tree depth
                "min_samples_split": [2, 5, 10, 15, 20],     # Minimum samples to split
                "min_samples_leaf": [1, 2, 4, 6, 8, 10],     # Minimum samples per leaf
                "max_features": [None, "sqrt", "log2", 0.5, 0.75, 1.0],  # Features per split
                "max_leaf_nodes": [None, 10, 20, 50, 100],   # Limit on leaf count
                "min_weight_fraction_leaf": [0.0, 0.05, 0.1] # Useful for imbalance
        }
    if model_type == 'gradientBoost':
        return {
                "n_estimators": [100, 200, 300],       # Number of boosting stages
                "learning_rate": [0.01, 0.05, 0.1],   # Step size shrinkage
                "max_depth": [3, 4, 5],               # Depth of individual trees
                "min_samples_split": [2, 5, 10],      # Min samples to split a node
                "min_samples_leaf": [1, 2, 4],        # Min samples per leaf
                "subsample": [0.8, 1.0],              # Use <1.0 for stochastic boosting
                "max_features": ["sqrt", "log2", None] # Features considered per split
        }
    if model_type == 'neuralNetwork':
        return {
            'layers': [[80, 100, 200, 100, 50, 1], [80, 50, 1]],
            'patience': [5, 10],
            'lr': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
            'batch_size': [16, 32]
        }
    if model_type == 'supportVectorMachine':
        return {
            "C": [0.01, 0.1, 1, 10, 100],           # Regularization strength
            "kernel": ["linear", "poly", "rbf", "sigmoid"],  # Kernel types
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1], # Kernel coefficient
            "degree": [2, 3, 4, 5],                 # For 'poly' kernel only
            "coef0": [0.0, 0.1, 0.5, 1.0],          # For 'poly' and 'sigmoid'
            "shrinking": [True, False],             # Use shrinking heuristic
            "max_iter": [100000],
            "class_weight": [None, "balanced"],     # Classification only (SVC)
        }
    if model_type == 'kmeans':
        return {
            'n_clusters': range(2, 21),
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300],
            'tol': [1e-3, 1e-4, 1e-5],
            'n_init': [5, 10, 15, "auto"],
            'verbose': [0]
        } 
    if model_type == 'knn':
        return {
                'k': list(range(1, 11))
        }
    if model_type == 'prox':
        return {
            'n_features':range(1,96),
            '_type':['mean','median'],
            '_sorting_metric':['same','variance']
        }

    if model_type == 'sgd':
        return {
            'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [100000],
            'tol': [1e-3, 1e-4],
            'l1_ratio': [0.15, 0.5, 0.85]
        }

def get_model_class(model_type:ModelTypes) -> type:
    if model_type == 'decisionTree':
        return DecisionTreeClassifier  
    if model_type == 'gradientBoost':
        return GradientBoostingClassifier
    if model_type == 'neuralNetwork':
        return NN
    if model_type == 'supportVectorMachine':
        return SVC
    if model_type == 'kmeans':
        return KmeansClassifier
    if model_type == 'knn':
        return KnnClassifier
    if model_type == 'sgd':
        return SGDClassifier

class Cross_validatior:
    def __init__(self, problems:list[str]|None, X:pd.DataFrame, Y:pd.DataFrame, cv:int) -> None:
        self.problems = problems
        self.X = X.drop(columns=['problem'])
        self.Y = Y
        if self.problems is not None:
            folds = self.split_problems(cv)
            fold_data = []
            for train_problems, validation_problems in folds:
                X_train = X[X['problem'].isin(train_problems)]
                X_train = X_train.drop(columns=['problem'])
                Y_train = Y[X['problem'].isin(train_problems)]

                X_validation = X[X['problem'].isin(validation_problems)]
                X_validation = X_validation.drop(columns=['problem'])
                Y_validation = Y[X['problem'].isin(validation_problems)]
                fold_data.append((X_train, X_validation, Y_train, Y_validation))
            self.cv_data = fold_data

    def split_problems(self,cv:int) -> list[tuple[list[str],list[str]]]: 
        assert self.problems is not None, "this should not happen"
        size = len(self.problems) // cv
        folds = []
        for f in range(cv):
            train_problems = self.problems.copy()
            validation_problems = []
            for i in reversed(list(range(size * f, size * (f+1)))):
                validation_problems.append(train_problems.pop(i))
            folds.append((train_problems, validation_problems))
        return folds

    def cross_validate(self, estimator:BaseEstimator) -> np.ndarray:
        if self.problems is None:
            return cross_val_score(estimator, self.X, self.Y)
        results = []
        for X_train, X_validation, Y_train, Y_validation in self.cv_data:
            _estimator = clone(estimator)
            _estimator.fit(X_train, Y_train)
            y_pred = _estimator.predict(X_validation)
            results.append(accuracy_score(Y_validation, y_pred))
        return np.array(results)

def re_arrange(data:dict) -> dict:
    solvers = list(data.keys())
    rearranged_data = {}
    for solver in solvers:
        for cores in data[solver]:
            for dp in data[solver][cores]:
                name = dp['name']
                model = dp['model']
                if not (model, name) in rearranged_data:
                    rearranged_data[model, name] = {'model':model, 'name':name, 'search':dp['search'], 'performances': {}}
                rearranged_data[model,name]['performances'][solver,cores] = {'time': dp['time'], 'obj': dp['objective'], 'has_solution': dp['has_solution'], 'optimal':dp['optimal']}
    return rearranged_data

def predict_used_alg(pred:list, performance_data:dict) -> tuple[str,str]:
    algs = [(str(sol), str(c)) for (sol, c) in pred if c > 0]
    _s = performance_data['search']
    perfs = performance_data['performances']
    if _s == 'Satisfy':
        best_algs = [a for a in algs if perfs[a]['has_solution']]
    elif _s == 'Maximise':
        not_none = [perfs[a]['obj'] for a in algs if perfs[a]['obj'] is not None and perfs[a]['has_solution']]
        max_val = max(not_none) if len(not_none) > 0 else None
        best_algs = [a for a in algs if perfs[a]['obj'] == max_val]
        optimal = [a for a in best_algs if perfs[a]['optimal'] == 'Optimal']
        if len(optimal) > 0:
            best_algs = optimal
    elif _s == 'Minimise':
        not_none = [perfs[a]['obj'] for a in algs if perfs[a]['obj'] is not None and perfs[a]['has_solution']]
        min_val = min(not_none) if len(not_none) > 0 else None
        best_algs = [a for a in algs if perfs[a]['obj'] == min_val]
        optimal = [a for a in best_algs if perfs[a]['optimal'] == 'Optimal']
        if len(optimal) > 0:
            best_algs = optimal
    else:
        raise Exception(f'Unknown search {_s}')
    if len(best_algs) == 0:
        return algs[0]
    alg = min([(a, perfs[a]['time']) for a in best_algs], key=lambda x: x[1])[0]
    return alg
