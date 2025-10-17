from typing import Literal
from numpy._core import numeric
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import argparse
import argcomplete
from argcomplete.completers import FilesCompleter
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from copy import deepcopy
import random

ModelTypes = Literal['decisionTree','gradientBoost', 'neuralNetwork', 'supportVectorMachine', 'kmeans', 'knn', 'prox', 'sgd']

class Closer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features:int, _type:Literal['mean','median'], _sorting_metric:Literal['same', 'variance']) -> None:
        self.n_features = n_features
        self._type = _type
        self._sorting_metric = _sorting_metric

    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        X_negative = X[y == 0]
        X_positive = X[y == 1]
        cols = X.columns
        cols_vals_positive = {}
        cols_vals_negative = {}
        for col in cols:
            _score_p = X_positive[col].mean() if self._type == "mean" else X_positive[col].median()
            _score_n = X_negative[col].mean() if self._type == "mean" else X_positive[col].median()
            cols_vals_positive[col] = (_score_p, X_positive[col].var() if self._sorting_metric == "variance" else np.mean([np.abs(_score_p - X_positive.iloc[i][col]) for i in range(len(X_positive))]))
            cols_vals_negative[col] = (_score_n, X_positive[col].var() if self._sorting_metric == "variance" else np.mean([np.abs(_score_n - X_positive.iloc[i][col]) for i in range(len(X_negative))]))
        self.positive = {k:v[0] for k,v in sorted(cols_vals_positive.items(), key=lambda x: x[1][1])[:self.n_features]}
        self.negative = {k:v[0] for k,v in sorted(cols_vals_negative.items(), key=lambda x: x[1][1])[:self.n_features]}

    def predict(self, X:pd.DataFrame):
        positive_results = []
        negative_results = []
        for i in range(len(X)):
            el = X.iloc[i]
            pos_arr = []
            neg_arr = []
            for col, v in self.positive.items():
                pos_arr.append((el[col] - v) ** 2)
            for col, v in self.negative.items():
                neg_arr.append((el[col] - v) ** 2)
            positive_results.append(sum(pos_arr) / len(pos_arr))
            negative_results.append(sum(neg_arr) / len(neg_arr))
        return pd.DataFrame([1 if positive_results[i] < negative_results[i] else 0 for i in range(len(positive_results))], columns=['y'])

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))

def set_seed(seed=42):
    random.seed(seed)                     # Python built-in random
    np.random.seed(seed)                  # Numpy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU (if using)
    torch.cuda.manual_seed_all(seed)      # All GPUs (if using multi-GPU)

    # For deterministic behavior (may affect performance)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

class NoneScaler:
    def __init__(self) -> None:
        pass
    def fit(*args) -> None:
        pass

    def transform(self, data):
        return data

class PCAPreprocessor:
    def __init__(self, pca_type:str) -> None:
        if pca_type == 'None':
            self.pca = None
        else:
            n_components = pca_type if pca_type == 'mle' else int(pca_type)
            self.pca = PCA(n_components=n_components)

    def fit(self, X:pd.DataFrame):
        if not self.pca:
            return
        self.pca.fit(X.select_dtypes(include=['number']))

    def transform(self, X:pd.DataFrame):
        if not self.pca:
            return X
        non_numeric_X = X.select_dtypes(exclude=['number'])
        numeric_X = X.select_dtypes(include=['number'])
        components = self.pca.transform(numeric_X)
        comp_names = [f'comp_{i}' for i in range(len(components[0]))]
        pca_df = pd.DataFrame(components, columns=comp_names, index=X.index)
        return pd.concat([non_numeric_X, pca_df], axis=1)

class KnnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k:int) -> None:
        self.k = k

    def fit(self, X, Y):
        self.X = X.to_numpy(dtype=float)
        self.Y = Y

    def predict(self, X:pd.DataFrame, t=False) -> np.ndarray:
        preds = []
        for x in X.to_numpy(dtype=float):
            x_dists = []
            for i, _x in enumerate(self.X):
                x_dists.append((np.sum((x - _x ) ** 2), i))
            x_dists = sorted(x_dists, key = lambda x: x[0])
            if t:
                print(x_dists[:self.k])
            y_vals = [self.Y.iloc[idx] for _, idx in x_dists[:self.k]]
            counts = (y_vals.count(0), y_vals.count(1))
            preds.append(np.argmax(counts))
        return np.array(preds)

class KmeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **hyperparams) -> None:
        self.model = KMeans(**hyperparams)

    def fit(self, X, Y):
        preds = self.model.fit_predict(X)
        cluster_vals = {}
        assert len(X) == len(Y)
        for i in range(len(preds)):
            if not preds[i] in cluster_vals:
                cluster_vals[preds[i]] = [0,0]
            cluster_vals[preds[i]][int(Y.iloc[i])] += 1
        self.labels = {cluster: np.argmax(vals) for cluster, vals in cluster_vals.items()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.array(list(map(lambda x: self.labels[x], preds)))

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))

class NN(BaseEstimator, ClassifierMixin):
    def __init__(self, layers: list[int], lr: float = 0.01, batch_size: int = 32, patience: int = 5):
        """
        layers: list of layer sizes
        lr: learning rate
        batch_size: mini-batch size
        patience: number of epochs to wait for improvement before stopping
        """
        self.layers = layers
        self.n_epochs = 10000
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self._build_model()

    def _build_model(self):
        nn_layers = []
        for i in range(1, len(self.layers)):
            nn_layers.append(nn.Linear(self.layers[i-1], self.layers[i]))
            if i < len(self.layers) - 1:  # activation except last layer
                nn_layers.append(nn.LeakyReLU())
        self.nn = nn.Sequential(*nn_layers)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, verbose=False):
        # Convert to numpy arrays
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values if isinstance(y, pd.DataFrame) else np.array(y)

        # Split 90% train, 10% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=0.1, stratify=y_np
        )

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Prepare DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.nn.parameters(), lr=self.lr)

        # Early stopping variables
        best_loss = float("inf")
        best_model = deepcopy(self.nn.state_dict())
        epochs_no_improve = 0

        # Training loop
        for epoch in range(self.n_epochs):
            self.nn.train()
            tot_loss = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.nn(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                tot_loss.append(float(loss.detach()))

            # Validation loss
            self.nn.eval()
            with torch.no_grad():
                val_outputs = self.nn(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            if verbose and epoch % 10 == 0:
                loss = np.mean(tot_loss)
                print(f"{epoch}/{self.n_epochs} - ({loss:.4f}, {val_loss:.4f})")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.nn.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                if verbose:
                    print("early stopping")
                break

        self.nn.load_state_dict(best_model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.nn.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        with torch.no_grad():
            logits = self.nn(torch.tensor(X_np, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.nn.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        with torch.no_grad():
            logits = self.nn(torch.tensor(X_np, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().flatten()
        return np.vstack([1 - probs, probs]).T

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))

class ContrastiveLearningFeatures:
    def __init__(self) -> None:
        self.nn = nn.Sequential(
            nn.Linear(95, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 90),
            nn.LeakyReLU(),
            nn.Linear(90, 80),
            nn.Tanh()
        )
        self.margin = 1
        self.lr = 4e-6
        self.batch_size = 32
        self.patience = 5

    def __loss(self, z1: torch.Tensor, z2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(z1, z2)
        loss_same = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_diff = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_same + loss_diff)

    def fit(self, X:pd.DataFrame, y:pd.DataFrame, problem_aware:bool):
        if problem_aware:
            problems = X['problem'].unique().tolist()
            train_problems, validation_problems = train_test_split(problems)
            X_train = X[X['problem'].isin(train_problems)].drop(columns=['problem']).values
            y_train = y[X['problem'].isin(train_problems)].values
            X_val = X[X['problem'].isin(validation_problems)].drop(columns=['problem']).values
            y_val = y[X['problem'].isin(validation_problems)].values
        else:
            _X = X.drop(columns=['problem']).values
            _y = y.values
            X_train, X_val, y_train, y_val = train_test_split(_X, _y, stratify=_y)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size // 2, shuffle=True
        )
        optimizer = optim.Adam(self.nn.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_model = deepcopy(self.nn.state_dict())
        epochs_no_improve = 0

        for epoch in range(1000):
            self.nn.train()
            tot_loss = []
            for X_batch, y_batch in train_loader:
                batch = X_batch.shape[0]
                _size = batch // 2
                x1 = X_batch[_size:]
                x2 = X_batch[:_size]
                y1 = y_batch[_size:]
                y2 = y_batch[:_size]
                if y1.shape[0] > y2.shape[0]:
                    y1 = y1[:-1]
                    x1 = x1[:-1]
                labels = torch.zeros_like(y1)
                labels[y1 == y2] = 1
                optimizer.zero_grad()
                z1 = self.nn(x1)
                z2 = self.nn(x2)
                loss = self.__loss(z1, z2, labels)
                loss.backward()
                optimizer.step()
                tot_loss.append(float(loss.detach()))

            self.nn.eval()
            with torch.no_grad():
                val_loss = []
                for X_batch, y_batch in val_loader:
                    batch = X_batch.shape[0]
                    _size = batch // 2
                    x1 = X_batch[_size:]
                    x2 = X_batch[:_size]
                    y1 = y_batch[_size:]
                    y2 = y_batch[:_size]
                    if y1.shape[0] > y2.shape[0]:
                        y1 = y1[:-1]
                        x1 = x1[:-1]
                    labels = torch.zeros_like(y1)
                    labels[y1 == y2] = 1
                    z1 = self.nn(x1)
                    z2 = self.nn(x2)
                    loss = self.__loss(z1, z2, labels)
                    val_loss.append(float(loss.detach()))

                val_loss = np.mean(val_loss)

            if epoch % 1 == 0:
                loss = np.mean(tot_loss)
                print(f"{epoch}/{10000} - ({loss:.4f}, {val_loss:.4f})")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.nn.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print("early stopping")
                break

            self.nn.load_state_dict(best_model)

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        self.nn.eval()
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        with torch.no_grad():
            representation = self.nn(torch.tensor(X_np, dtype=torch.float32))
            np_representation = representation.detach().numpy()
            return pd.DataFrame(np_representation, index=X.index)

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
    if model_type == 'prox':
        return Closer
    if model_type == 'sgd':
        return SGDClassifier

def get_scaler(scaler_type:Literal['standard', 'minMax', 'None']) -> type:
    if scaler_type == 'standard':
        return StandardScaler
    if scaler_type == 'minMax':
        return MinMaxScaler
    return NoneScaler

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

def is_valid_type(pca_type:str) -> bool:
    if pca_type == 'None' or pca_type == 'mle':
        return True
    try:
        int(pca_type)
        return True
    except:
        return False

def main(args):
    data_file = args.data
    scaler_name = args.scaler
    model_name = args.model
    test_size = args.test_size
    cv = args.cv
    random_seed = args.random_seed
    problem_unaware = args.problem_unaware
    pca_type = args.pca

    set_seed(random_seed)

    data = pd.read_csv(data_file)
    problems = data['problem'].unique().tolist()
    train_problems = None

    if problem_unaware:
        train_problems, test_problems = train_test_split(problems, test_size=test_size)
        train_data = data[data['problem'].isin(train_problems)]
        test_data = data[data['problem'].isin(test_problems)]
    else:
        train_data, test_data = train_test_split(data, test_size=test_size)

    if not is_valid_type(pca_type):
        print("wrong pca input, use --help for help")
        return

    train_data = train_data.drop(columns=['name'])
    test_data = test_data.drop(columns=['name'])
    y_train = train_data['y']
    x_train = train_data.drop(columns=['y'])
    y_test = test_data['y']
    x_test = test_data.drop(columns=['y'])

    if scaler_name == 'std':
        scaler_name = 'standard'
    elif scaler_name == 'mm':
        scaler_name = 'minMax'
    scaler_class = get_scaler(scaler_name)
    scaler = scaler_class()
    numeric_cols = x_train.select_dtypes(include=['number']).columns
    scaler.fit(x_train[numeric_cols])
    x_train[numeric_cols] = scaler.transform(x_train[numeric_cols])
    x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    if False:
        contrastive = ContrastiveLearningFeatures()
        contrastive.fit(x_train, y_train, not problem_unaware)
        non_numeric_x_train = x_train.select_dtypes(exclude=['number'])
        non_numeric_x_test = x_test.select_dtypes(exclude=['number'])
        x_train_feat = contrastive.transform(x_train[numeric_cols])
        x_test_feat = contrastive.transform(x_test[numeric_cols])
        x_train = pd.concat([non_numeric_x_train, x_train_feat], axis=1)
        x_test = pd.concat([non_numeric_x_test, x_test_feat], axis=1)

    pca_preprocessor = PCAPreprocessor(pca_type)
    pca_preprocessor.fit(x_train)
    x_train = pca_preprocessor.transform(x_train)
    x_test = pca_preprocessor.transform(x_test)

    if model_name == 'dt':
        model_name = 'decisionTree'
    elif model_name == 'gb':
        model_name = 'gradientBoost'
    elif model_name == 'nn':
        model_name = 'neuralNetwork'
    elif model_name == 'svm':
        model_name = 'supportVectorMachine'
    elif model_name == 'km':
        model_name = 'kmeans'
    elif model_name == 'kn':
        model_name = 'knn'

    model_class = get_model_class(model_name)
    model_hyperparams = get_hyperparameters(model_name)

    hyperparams = list(ParameterGrid(model_hyperparams))

    scores = []

    cross_validate = Cross_validatior(train_problems, x_train, y_train, cv=cv)
    for hyperparam in tqdm(hyperparams):
        model = model_class(**hyperparam)
        score = cross_validate.cross_validate(model)	
        scores.append(np.mean(score))

    best_score_idx = np.argmax(scores)

    best_parameters = hyperparams[best_score_idx]
    print("best found hyperparams:", best_parameters)
    print("with score:", scores[best_score_idx])
    model = model_class(**best_parameters)
    numeric_cols = x_train.select_dtypes(include=['number']).columns
    model.fit(x_train[numeric_cols], y_train)
    y_pred = model.predict(x_test[numeric_cols])
    accuracy = accuracy_score(y_test, y_pred)
    count = (list(y_train).count(0), list(y_train).count(1))
    majority = np.zeros_like if count[0] > count[1] else np.ones_like
    majority_accuracy = accuracy_score(y_test, majority(y_test))
    print("test set accuracy:", accuracy)
    print("majority accuracy:", majority_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='fit_ml',
                    description='fit a classifier that predicts if a solver will score better with parallelisation enabled.')
    parser.add_argument('-m', '--model', type=str, choices=['decisionTree', 'dt', 'gradientBoost', 'gb', 'neuralNetwork','nn', 'supportVectorMachine', 'svm', 'kmeans', 'km','knn', 'kn', 'prox', 'sgd'], required=True, help='The model to fit.')
    parser.add_argument('-s', '--scaler', type=str, choices=['standard', 'std', 'minMax', 'mm', 'None'], required=False, default='std', help='How to scale the data. None does not scale it.')
    parser.add_argument('-t', '--test-size', type=float, required=False, default=.2, help='The amount of data to reserve to the test process. default to 20%%.')
    parser.add_argument('-c', '--cv', type=int, required=False, default=5, help='Number of cross-validation steps to perform.')
    parser.add_argument('-r', '--random-seed', type=int, required=False, default=42, help='Random seed to use.')
    parser.add_argument('-pu', '--problem-unaware',  action='store_false', help='If the validation and test set should be problem aware.')
    parser.add_argument('-p', '--pca', type=str, default='None', help='Whether or not to apply PCA decomposition. If None then nothing happens, otherwise it must contain the number of components to decompose to or mle for maximum likelyhood decomposition. Default None.')
    parser.add_argument('-d', '--data', type=str, required=True, help='The .csv file that contains the data to use.').completer = FilesCompleter(allowednames=["*.csv"])

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(args)
