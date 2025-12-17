import shutil, sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
import json
from uuid import uuid4

data = pd.read_csv(sys.argv[1])
rnd_state = int(sys.argv[2])
problems = data['problem'].unique().tolist()

if sys.argv[3] == 'pu':
    train_problems, test_problems = train_test_split(problems, test_size=.2, random_state=rnd_state)
    train_data = data[data['problem'].isin(train_problems)]
    test_data = data[data['problem'].isin(test_problems)]
else:
    train_data, test_data = train_test_split(data, test_size=.2, random_state=rnd_state)

train_data = train_data.drop(columns=['name'])
test_data = test_data.drop(columns=['name', 'problem'])
y_train = train_data['y']
x_train = train_data.drop(columns=['y'])
y_test = test_data['y']
x_test = test_data.drop(columns=['y'])

x_train = x_train.drop(columns=['problem'])
name = f'/tmp/autosklearn_interpretable_models_example_tmp_{sys.argv[1].split(".")[0]}_{rnd_state}_{str(uuid4())}'
shutil.rmtree(name, ignore_errors=True)

automl = AutoSklearnClassifier(
    n_jobs=1,
    memory_limit=5000,
    time_left_for_this_task=60*60,
    tmp_folder=name,
    include={
        "classifier": ["decision_tree", "lda", "sgd"],
        "feature_preprocessor": [
            "no_preprocessing",
            "polynomial",
            "select_percentile_classification",
        ],
    },
    seed=rnd_state,
    resampling_strategy="holdout-iterative-fit",
    ensemble_kwargs={"ensemble_size": 1},
)

automl.fit(x_train, y_train, X_test=x_test, y_test=y_test, dataset_name=sys.argv[1].split('/')[-1].replace('.csv',''))

predictions = automl.predict(x_test)
test_accuracy = accuracy_score(y_test, predictions)
train_accuracy = accuracy_score(y_train, automl.predict(x_train))
count = (list(y_train).count(0), list(y_train).count(1))
majority = np.zeros_like if count[0] > count[1] else np.ones_like
majority_accuracy = accuracy_score(y_test, majority(y_test))
print(json.dumps({"train_score": train_accuracy, "test_accuracy": test_accuracy, "majority_accuracy": majority_accuracy}))
