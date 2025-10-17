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
    * **datasets**: a sub-folder with all the csv files that can be used to train the ml models. they are named as ```solver_c1c2.csv``` where c1 and c2 are number of cores and c1 < c2. Each csv has all the values of ```features.csv``` plus a y column (0 / 1) for preduction.
    * **figures**: a sub-folder with all the generated figures.
    * **minizinc_instances**: a sub-folder with all the instances and model used in the projects.


## Reproducibility

To reproduce our experiments you can simply run the ```fit_ml.py``` script with a dataset. 
Before that, it is necessary to install all the libraries:
```
pip install -r requirements.txt
```
Then run the script as:
```
python fit_ml.py -m <ml model> -d <csv dataset>
```
to see all the available options use the help command:
```
python fit_ml.py -h
```
or:
```
python fit_ml.py --help
```