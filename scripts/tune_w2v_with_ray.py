import os
import argparse
import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from recsys.models import tune_w2v
from recsys.utils import pickle_save, absolute_filename

RAY_CLUSTER_ADDRESS = os.getenv("RAY_CLUSTER_ADDRESS") # For use in CDSW/CML

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", 
    help="Directory name for HPO experiment results.",
    required=False
)
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
parser.add_argument(
    "--ray-address",
    help="Address of Ray cluster for seamless distributed execution.",
    required=False,
)
parser.add_argument(
    "-cml",
    help="Set this flag if using CDSW or CML for seamless distributed execution",
    action="store_true"
)
parser.add_argument(
    "--asha",
    help="Enable an ASHA Scheduler to stop underperforming trials early during hyperparameter sweep",
    action="store_true"
)
args, _ = parser.parse_known_args()


# If necessary, connect to an existing Ray Cluster for distributed execution
if args.ray_address: 
    ray.init(address=args.ray_address)
if args.cml:
    ray.init(address=RAY_CLUSTER_ADDRESS)

# Define the hyperparameter search space for Word2Vec algorithm
search_space = {
    "dataset": "ecomm",
    "k": 10,
    #"size": tune.grid_search(list(np.arange(10,106, 6))),
    #"window": tune.grid_search(list(np.arange(1,22, 3))),
    #"ns_exponent": tune.grid_search(list(np.arange(-1, 1.2, .2))),
    #"alpha": tune.grid_search([0.001, 0.01, 0.1]),
    "negative": tune.grid_search(list(np.arange(1,22, 3))),
    "iter": 10,
    "min_count": 1,
    "workers": 6,
    "sg": 1,
}

# The ASHA Scheduler will stop underperforming trials in a principled fashion
asha_scheduler = ASHAScheduler(max_t=100, grace_period=10) if args.asha else None

# Set the stopping critera -- use the smoke-test arg to test the system 
stopping_criteria = {"training_iteration": 1 if args.smoke_test else 9999}

# Perform hyperparamter sweep with Ray Tune
analysis = tune.run(
    tune_w2v,
    name=args.name,
    local_dir=absolute_filename("ray_results"),
    metric="recall_at_k",
    mode="max",
    scheduler=asha_scheduler,
    stop=stopping_criteria,
    num_samples=1,
    verbose=1,
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    config=search_space,
)
print("Best hyperparameters found were: ", analysis.best_config)

"""
# Plot all trials as a function of epochs
dfs = analysis.trial_dataframes
ax = None
for d in dfs.values():
    ax = d.recall_at_k.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs");
ax.set_ylabel("Recall@10");
"""
