import argparse
import numpy as np
import matplotlib.pyplot as plt 

from ray.tune import Analysis

from recsys.data import load_ecomm, train_test_split
from recsys.models import train_w2v, RecallAtKLogger, LossLogger
from recsys.metrics import recall_at_k, mrr_at_k
from recsys.utils import absolute_filename, pickle_save

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    help="Directory for HPO experiment results -- providing this will result in a W2V model \
          trained with the best hyperparameters from that experiment. (Note: if not provided, \
          default W2V hyperparameters are used instead. These can be modified directly in this script."
)
parser.add_argument(
    "-k", default=10,
    help="Number of recommendations to generate for model evaluation. Default is 10."
)
parser.add_argument(
    "--outdir",
    help="Directory in which to save trained model embeddings and training metrics. Default is `output/`",
    default=absolute_filename("output/")
)
args = parser.parse_known_args()


# load data
sessions = load_ecomm()
train, test, valid = train_test_split(sessions, test_size=1000)

# determine word2vec parameters to train with
if args.name:
    analysis = Analysis(absolute_filename("ray_results", args.name), 
                        default_metric="recall_at_k",
                        default_mode="max")

    w2v_params = analysis.get_best_config()
else:
    # These the few required parameters for training Word2Vec for this use case. 
    # All other parameters will rely on Gensim defaults.  
    w2v_params = {
        "min_count": 1,
        "iter": 5,
        "workers": 10,
        "sg": 1,
    }

# Instantiate callback to measurs Recall@K on the validation set after each epoch of training
ratk_logger = RecallAtKLogger(valid, k=args.k, save_model=True)
# Instantiate callback to compute Word2Vec's training loss on the training set after each epoch of training
loss_logger = LossLogger()
# Train Word2Vec model and retrieve trained embeddings
embeddings = train_w2v(train, w2v_params, [ratk_logger, loss_logger])

# Save results
pickle_save(ratk_logger.recall_scores, absolute_filename(args.outdir, f"recall@k_per_epoch.pkl"))
pickle_save(loss_logger.training_loss, absolute_filename(args.outdir, f"trainloss_per_epoch.pkl"))

# Save trained embeddings
embeddings.save(absolute_filename(args.outdir, f"embeddings.wv"))

# Visualize metrics as a function of epoch
plt.plot(np.array(ratk_logger.recall_scores)/np.max(ratk_logger.recall_scores))
plt.plot(np.array(loss_logger.training_loss)/np.max(loss_logger.training_loss))
plt.show()

# Print results on the test set
print(recall_at_k(test, embeddings, k=args.k))
print(mrr_at_k(test, embeddings, k=args.k))

