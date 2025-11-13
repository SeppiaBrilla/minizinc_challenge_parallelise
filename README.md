This is the repository for the parallelise CP solver project. The project structure is:
* **requirements.txt**: the required libraries for reproduciability. install them with the command ``` pip install -r requirements.txt ```.
* **images.ipynb**: the python notebook used to generate the images. 
* **fit_ml.py**: a python script to train the machine learning models to predict if to parallelise or not.
* **data**: a folder containing all the data necessary. It contains:
    * **results.json**: a json file with all the statistics for each instance. The json is this structured:
        ```json
        {
            "solver":{
                "n_cores":[
                    {
                        "name": "instance name",
                        "model": "instance model",
                        "time": "solving time in ms",
                        "objective": "objective value if present, null otherwise",
                        "search": "type of search: Maximise, Minimise or Satisfy",
                        "optimal": "If the solution is Optimal, Unsat or if its state is Unknown"
                    },
                    "..."
                ],
                "..."
            },
            "..."
        }
    * **features.csv**: a csv file with all the fzn2feat instance features.
    * **all_options.json**: a json file with all portfolio options under the 8 cores trashold.
    * **comparison_2024.json**: a json file with the scores of all portfolios on the 2024 minizinc competition. There is a key for each portfolio and it contains:
        * **w**: the list of all portfolio agaist which the current portfolio won.
        * **l**: the list of all portfolio agaist which the current portfolio has lost.
        * **w**: the list of all portfolio agaist which the current portfolio had the same performance.
    * **comparison_2025.json**: a json file with the scores of all portfolios on the 2025 minizinc competition. There is a key for each portfolio and it contains:
        * **w**: the list of all portfolio agaist which the current portfolio won.
        * **l**: the list of all portfolio agaist which the current portfolio has lost.
        * **w**: the list of all portfolio agaist which the current portfolio had the same performance.
    * **datasets**: a sub-folder with all the csv files that can be used to train the ml models. they are named as ```solver_c1c2.csv``` where c1 and c2 are number of cores and c1 < c2. Each csv has all the values of ```features.csv``` plus a y column (0 / 1) for preduction.
    * **figures**: a sub-folder with all the generated figures.
    * **minizinc_instances**: a sub-folder with all the instances and model used in the projects.
    * **ml_results**: a sub-folder containing a set of json file with the results of our experiments on classification problems. The file names are formatted as *MLmodel_minMaxing_rndSeed_dataset_TestType*. 
* **algorithm_selection**: a folder with all the python code used for this project. ```classification.py``` is the script used to train the machine learning models and the classification folder contains all the custom machine learning models. All other scripts (but helper, scoring and scalers) are useful for portfolio comparisons. Please use ```python <portfolio-script>``` for more information.


## Reproducibility

To reproduce our experiments you can simply run the ```algorithm_selection/classification.py``` script with a dataset. 
Before that, it is necessary to install all the libraries:
```
pip install -r requirements.txt
```
Then run the script as:
```
python algorithm_selection/classification.py -m <ml model> -d <csv dataset>
```
to see all the available options use the help command:
```
python algorithm_selection/classification.py -h
```
or:
```
python algorithm_selection/classification.py --help
```