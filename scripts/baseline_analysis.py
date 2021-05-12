from recsys.data import load_ecomm, train_test_split
from recsys.models import association_rules_baseline
from recsys.metrics import recall_at_k_baseline, mrr_at_k_baseline

# load data
sessions = load_ecomm()
train, test, valid = train_test_split(sessions, test_size=1000)

# Construct a co-occurrence matrix containing how frequently 
# each item is found in the same session as any other item
comatrix = association_rules_baseline(train)

# Recommendations are generated as the top K most frequently co-occurring items
# Compute metrics on these recommendations for each (query item, ground truth item)
# pair in the test set
recall_at_10 = recall_at_k_baseline(test, comatrix, k=10)
mrr_at_10 = mrr_at_k_baseline(test, comatrix, k=10)

print("Recall@10:", recall_at_10)
print("MRR@10:", mrr_at_10)