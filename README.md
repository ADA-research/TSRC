# Official Implementation of the TSRC Paper

[![CC BY 4.0][cc-by-shield]][cc-by]

This repository contains the official implementation of the paper **Time Series Representations Classroom (TSRC): A
Teacher-Student-based Framework for Interpretability-enhanced Unsupervised Time Series Representation Learning**.

Current status: submitted for journal publication (under review).

# Requirements

Please check the `requirements.txt` file for the required packages.

# Datasets

The datasets used in the paper are the UCR datasets. You can download them
from [here](http://www.timeseriesclassification.com/Downloads/).
After downloading the datasets, create a folder named `data` in the root directory of the project and put
the datasets in it, preferably name it `ucr_classification_univariate`.
We listed the names of the datasets we used in `experiments/ucr_classification_univariate_dataset_names_to_run.txt`.

# Reproducing the results presented in the paper

The runner scripts are found in `runners/py`. Every running scripts has the parameters described in the same file.

To reproduce the results in the paper, you can do the following:

1. Get the representations of the baselines models. There is a file for each baseline model considered (
   check `runners/py`
   for the runner scripts).
   You can pass the parameters of the baseline models to the corresponding runner script as described in the paper.
2. Cluster the representations of the baselines models.
   You can do that by running the `runner_cluster_and_evaluate_representations.py` script and passing the path to the
   representations of the baselines models.
3. Get the representations of the TSRC framework.
   You can pass the parameters of the TSRC framework to the `runner_extract_representations_tsrc.py` script as described
   in the paper.
4. Cluster the representations of the TSRC framework.
   You can do that by running the `runner_cluster_and_evaluate_representations.py` script and passing the path to the
   representations of the TSRC framework.
5. You will then have the results of the clustering in the folder you specified in the runner script.
6. We provide a Jupyter Notebook to visualize and analyse the results in `notebooks/results_analysis.ipynb`. Please
   modify the paths
   in the notebook to match the paths of the results in your machine then hit run all button.

# Additional Experiments

If you would like to use the framework in your own experiments, you can import the class `TSRC` which can be found
in `tsrc` folder (check `runners/py/runner_extract_representations_tsrc.py` for a concrete and detailed example).

It is also possible to add more choices for student and teacher models by implementing them in a similar fashion as
current
models
(check `experiments/baselines` for examples) and adding them to the dictionaries in `tsrc/models_factory.py` file).

# Notes

* Please make sure to clone with the submodules to get the required packages for the project. You can do that by running
  the following command:

```
git clone --recurse-submodules
```

* Please make sure to add the path to the project and the path to the ts2vec dependency (`dependencies/ts2vec`) to the
  variable `PYTHONPATH` in your environment variables.

# Citation

If you use this code, please cite the following paper:

```
To be added
```

# Licence

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
