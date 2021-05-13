# Session-based Recommender Systems

This repo accompanies the Cloudera Fast Forward report [Session-based Recommender Systems](https://session-based-recommenders.fastforwardlabs.com/). It provides small library to train Word2Vec as a means of learning product or item representations in the context of user sessions (browsing histories, transaction histories, music playlists, etc.). These dense representations can then be used for item recommendation.  We formulate this under the Next Event Prediction task, that is, given a user's recent interaction, predict the next item they interact with (click on, purchase, listen to, etc.). 

Instructions are given both for general use (on a laptop, say), and for Cloudera CML and CDSW. We'll first describe what's here, then go through how to run everything.

## Structure
```
.
├── data        # This folder contains starter data.
├── scripts     # This contains scripts for *doing* things -- training models, analysing results.
├── notebooks   # This contains Jupyter notebooks that accompany the report and demonstrate basic usage.
└── recsys      # A small library of useful functions.
```
Let's examine each of the important folders in turn.


### `recsys`
```
├── data.py     # Contains functions for loading and processing data into sessions 
├── metrics.py  # Contains metrics for evaluation
├── models.py   # Contains wrappers for training Word2Vec both alone and with Ray Tune
└── utils.py    # Helper functions for serialization and I/O
```


### `scripts`
```
├── baseline_analysis.py     
├── setup_ray_cluster.py  
├── train_w2v_with_logging.py 
└── tune_w2v_with_ray.py    
```
An overview of what each of these scripts does is discussed below. 

### `notebooks`
```
├── Analyze_HPO_results.ipynb
└── Explore_Online_Retail_Dataset.ipynb    
```
These notebooks provide additional exploration and analysis. Please note that `Analyze_HPO_results.ipynb` is expressly for demonstration purposes as HPO output results explored within are not included in this repo. 

## Learning representations for session-based recommendations
To go from a fresh clone of the repo to the final state, follow these instructions in order.

### Installation
The code and applications within were developed against Python 3.8.8, and are likely also to function with more recent versions of Python.

To install dependencies, first create and activate a new virtual environment through your preferred means, then pip install from the requirements file. I recommend:

``` 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In CML or CDSW, no virtual env is necessary. Instead, inside a Python 3 session (with at least 2 vCPU / 4 GiB Memory), simply run

```
!pip3 install -r requirements.txt     # notice `pip3`, not `pip`
```

Note: if your session has an older Python image (3.6) use the alternative `requirements3.6.txt`:
```
!pip3 install -r requirements3.6.txt 
```

### Data

While we explored several datasets (and code exists in `recsys/data.py` to interact with those datasets), the analysis in this repo is focused on the [Online Retail](https://www.kaggle.com/vijayuv/onlineretail) dataset. This dataset is open source though you will need to create an account on Kaggle before downloading the data. In this repo we include a version of this dataset post-processed into customer sessions. These sessions represent all customer transactions from a UK-based online boutique selling specialty gifts collected between 12/01/2010 and 12/09/2011. In total there are purchase histories for 4,372 customers and 3,684 unique products. 

### Model training and analysis

The `scripts` directory contains scripts to train models in various formats and analyze results. Here we provide a high-level overview: 

* `scripts/baseline_analysis.py`: a common baseline for recommendation systems is to simply recommend the most popular items. This script computes the "Association Rules" baseline which considers how frequently each item co-occurrs with all other items in a session for each session in the training set. 
* `scripts/train_w2v_with_logging.py`: This script trains Gensim's implementation of the Word2Vec algorithm to learn representations for each item in a session. Identifying "similar" items then serves as the method for generating recommendations. Includes callbacks for monitoring metrics (Recall@K, training loss) as a function of training time (epochs). 
* `scripts/tune_w2v_with_ray.py`: The Word2Vec algorithm has a large hyperparameter space and the default values are subpar for the task of generating good item representations for recommendation systems. This scripts performs hyperparameter optimization (HPO) with [Ray Tune](https://docs.ray.io/en/master/tune/index.html). 
* [CDSW/CML only] `setup_ray_cluster.py`:  Hyperparameter optimization can be computationally expensive but this expense can be mitigated, in part, through distribution. This script initializes (and tears down) a Ray Cluster for distributed hyperparameter optimization. If using, follow the instructions in this script to setup the cluster, then run `tune_w2v_with_ray.py` with the appropriate arguments, and finally shutdown the cluster after HPO is complete. 


These scripts are not intended to be run in any particular order (with the exception noted above). Instead, they provide functionality for different use cases. To run scripts, follow this procedure in the terminal or a Session with at least 1vCPUs and 2GiBs of memory:

```
!python3 scripts/baseline_analysis.py
!python3 scripts/train_w2v_with_logging.py      # see optional arguments
!python3 scriptstune_w2v_with_ray.py            # see optional arguments for distributed HPO     
```

