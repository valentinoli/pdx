# Clustering estrogen receptor positive breast cancer tumors based on hormonal response type

Project in Machine Learning (CS-433)
EPFL, 2019

[Read the report](https://github.com/valentinoli/pdx/raw/master/report/report.pdf)

## Authors
* Lisa Dratva, [lisa.dratva@epfl.ch](mailto:lisa.dratva@epfl.ch)
* Michal Pleskowicz, [michal.pleskowicz@epfl.ch](mailto:michal.pleskowicz@epfl.ch)
* Valentin Oliver Loftsson, [valentin.loftsson@epfl.ch](mailto:valentin.loftsson@epfl.ch)

## Abstract
We employ unsupervised machine learning techniques to cluster subtypes of the most common breast cancer variant. Our results facilitate more targeted treatment of patients, responding to the urgent need for personalized medicine to treat breast cancer.

## Architecture
* [`scripts/`](scripts) directory contains all the code
* [`constants.py`](scripts/constants.py) includes constants used throughout the project
* [`helpers.py`](scripts/helpers.py) includes various helper functions
* [`load.py`](scripts/load.py) includes functions for loading and manipulating the data
* [`plots.py`](scripts/plots.py) includes plotting functions
* [`clustering_helpers.py`](scripts/clustering_helpers.py) includes helper functions used for the cluster analysis
* [`data_analysis.ipynb`](scripts/data_analysis.ipynb) is the Jupyter Notebook file that includes the exploratory data analysis and visualizations
* [`cluster_analysis.ipynb`](scripts/cluster_analysis.ipynb) is the Jupyter Notebook file that includes the cluster analysis

## Dependencies
* Data handling and ML libraries:
    * [`sklearn`](https://scikit-learn.org/stable/) (v. 0.21.3)
    * [`pandas`](https://pandas.pydata.org/) (v. 0.25.1)
    * [`numpy`](https://numpy.org/) (v. 1.16.5)
* Plotting libraries:
    * [`matplotlib`](https://matplotlib.org/)
    * [`seaborn`](https://seaborn.pydata.org/)
    * [`plotly` and `chart_studio`](https://help.plot.ly/)

## Reproduction
1. Install [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)
2. Install the abovementioned libraries
3. Run `data_analysis.ipynb` to reproduce results from the data analysis
4. Run `cluster_analysis.ipynb` to reproduce the clustering results