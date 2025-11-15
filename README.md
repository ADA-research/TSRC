# Official Implementation of the TSRC Paper

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of the paper **Time Series Representations Classroom (TSRC): A Teacher-Student-based Framework for Interpretability-enhanced Unsupervised Time Series Representation Learning**.

Authors: [Wadie Skaf](https://skaf.me), Mitra Baratchi, and Holger Hoos

The paper is published in the Machine Learning journal and can be accessed here: [Link](https://link.springer.com/article/10.1007/s10994-025-06895-x)

> 📌 If you find this repository helpful or insightful, please consider **giving it a star ⭐**. If you use it in your research, please **cite our paper** (see [Citation](#citation)).

# Requirements

Please check the `environment.yaml` file for the required packages.

# Datasets

The datasets used in the paper are the UCR datasets. You can download them from [here](http://www.timeseriesclassification.com/Downloads/). After downloading the datasets, create a folder named `data` in the root directory of the project and put the datasets in it, preferably name it `ucr_classification_univariate`. We listed the names of the datasets we used in `experiments/ucr_classification_univariate_dataset_names_to_run.txt`.

# Reproducing the results presented in the paper

The runner scripts are found in `runners/py`. Every running script has the parameters described in the same file.

## Raw Data

To reproduce the results of raw data in the paper:

1. Process the raw data and make them ready for further analysis. You can do that by running the `runners/py/runner_data_preprocessing.py` script. The script will also save the processed datasets in the folder you specified in the script.
2. To get the clustering results on raw data, run the script `runners/py/runner_cluster_and_evaluate_raw_data.py`. The script will save the clustering results in the folder you specified in the script.

## Representations

To reproduce the results related to representations in the paper, the first two steps are the following:

1. Get the representations of the baseline models. There is a file for each baseline model considered (check `runners/py` for the runner scripts). You can pass the parameters of the baseline models to the corresponding runner script as described in the paper.
2. Get the representations of the TSRC framework. You can pass the parameters of the TSRC framework to the `runner_extract_representations_tsrc.py` script as described in the paper.

Then you can run both clustering and classification experiments on the representations of the baseline models and the TSRC framework.

### Clustering

1. Cluster the representations of the baseline models. You can do that by running the `runner_cluster_and_evaluate_representations.py` script and passing the path to the representations of the baseline models.
2. Cluster the representations of the TSRC framework.
   You can do that by running the `runner_cluster_and_evaluate_representations.py` script and passing the path to the representations of the TSRC framework.
3. You will then have the results of the clustering in the folder you specified in the runner script.
4. We provide a Jupyter Notebook to visualize and analyze the results in `notebooks/results_analysis.ipynb`. Please modify the paths in the notebook to match the paths of the results in your machine, then press the "run all" button (also, include the raw data results files as the notebook will compare the clustering results of the raw data with the clustering results of the representations).

### Classification

1. Classify the representations of the baseline models.
   You can do that by running the `runner_classify_and_evaluate_representations.py` script and passing the path to the representations of the baseline models.
2. Classify the representations of the TSRC framework. You can do that by running the `runner_classify_and_evaluate_representations.py` script and passing the path to the representations of the TSRC framework.
3. You will then have the results of the classification in the folder you specified in the runner script.
4. We provide a Jupyter Notebook to visualize and analyze the results in `notebooks/results_analysis_classification.ipynb`. Please modify the paths in the notebook to match the paths of the results in your machine, then press the "run all" button.

# Additional Experiments

If you would like to use the framework in your own experiments, you can import the class `TSRC` which can be found in `tsrc` folder (check `runners/py/runner_extract_representations_tsrc.py` for a concrete and detailed example).

It is also possible to add more choices for student and teacher models by implementing them in a similar fashion as current models (check `experiments/baselines` for examples) and adding them to the dictionaries in `tsrc/models_factory.py` file.

# Notes

- Please make sure to clone with the submodules to get the required packages for the project. You can do that by running the following command:

```
git clone --recurse-submodules
```

- Please make sure to add the path to the project and the path to the `ts2vec` dependency (`dependencies/ts2vec`) to the variable `PYTHONPATH` in your environment variables.

## Citation

If you use this repository, adapt any of its components, or find it helpful for your work, please cite:

```bibtex
@article{skaf-2025-tsrc,
  title={Time Series Representations Classroom ({TSRC}): A Teacher-Student-based Framework for Interpretability-enhanced Unsupervised Time Series Representation Learning},
  author={Skaf, Wadie and Baratchi, Mitra and Hoos, Holger},
  journal={Machine Learning},
  year={2025},
  doi={10.1007/s10994-025-06895-x}
}
```

If this repository is useful to you, a star ⭐ is also very much appreciated and helps others discover the project.

# License

This work is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
