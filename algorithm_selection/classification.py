from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid
import argparse
import argcomplete
from argcomplete.completers import FilesCompleter
from helper import set_seed, get_hyperparameters, get_model_class, Cross_validatior
from scalers import get_scaler

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
    problem_aware = args.problem_aware
    json_out = args.json_output

    set_seed(random_seed)

    data = pd.read_csv(data_file)
    problems = data['problem'].unique().tolist()
    train_problems = None

    if problem_aware:
        train_problems, test_problems = train_test_split(problems, test_size=test_size)
        train_data = data[data['problem'].isin(train_problems)]
        test_data = data[data['problem'].isin(test_problems)]
    else:
        train_data, test_data = train_test_split(data, test_size=test_size)

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

    model = model_class(**best_parameters)
    numeric_cols = x_train.select_dtypes(include=['number']).columns
    model.fit(x_train[numeric_cols], y_train)
    y_pred = model.predict(x_test[numeric_cols])
    accuracy = accuracy_score(y_test, y_pred)
    count = (list(y_train).count(0), list(y_train).count(1))
    majority = np.zeros_like if count[0] > count[1] else np.ones_like
    majority_accuracy = accuracy_score(y_test, majority(y_test))
    if not json_out:
        print("best found hyperparams:", best_parameters)
        print("with score:", scores[best_score_idx])
        print("test set accuracy:", accuracy)
        print("majority accuracy:", majority_accuracy)
    else:
        import json
        print(json.dumps({"best_hyperparameters": best_parameters, "train_score": scores[best_score_idx], "test_accuracy": accuracy, "majority_accuracy": majority_accuracy}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='fit_ml',
                    description='fit a classifier that predicts if a solver will score better with parallelisation enabled.')
    parser.add_argument('-m', '--model', type=str, choices=['decisionTree', 'dt', 'gradientBoost', 'gb', 'neuralNetwork','nn', 'supportVectorMachine', 'svm', 'kmeans', 'km','knn', 'kn', 'sgd'], required=True, help='The model to fit.')
    parser.add_argument('-s', '--scaler', type=str, choices=['standard', 'std', 'minMax', 'mm', 'None'], required=False, default='std', help='How to scale the data. None does not scale it.')
    parser.add_argument('-t', '--test-size', type=float, required=False, default=.2, help='The amount of data to reserve to the test process. default to 20%%.')
    parser.add_argument('-c', '--cv', type=int, required=False, default=5, help='Number of cross-validation steps to perform.')
    parser.add_argument('-r', '--random-seed', type=int, required=False, default=42, help='Random seed to use.')
    parser.add_argument('-pa', '--problem-aware',  action='store_false', help='If the validation and test set should be problem aware.')
    parser.add_argument('-d', '--data', type=str, required=True, help='The .csv file that contains the data to use.').completer = FilesCompleter(allowednames=["*.csv"])
    parser.add_argument('-j', '--json-output', action='store_true', help='The if to use or not json as output format.')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(args)
