import collections
import numpy as np


def recall_at_k(test, embeddings, k: int = 10) -> float:
    """
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    ratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            ratk_score += 1
    ratk_score /= len(test)
    return ratk_score


def recall_at_k_baseline(test, comatrix, k: int = 10) -> float:
    """
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    ratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        try:
            co_occ = collections.Counter(comatrix[query_item])
            items_and_counts = co_occ.most_common(k)
            recommendations = [item for (item, counts) in items_and_counts]
            if ground_truth in recommendations: 
                ratk_score +=1
        except:
            pass
    ratk_score /= len(test)
    return ratk_score


def hitratio_at_k(test, embeddings, k: int = 10) -> float:
    """
    Implemented EXACTLY as was done in the Hyperparameters Matter paper. 
    In the paper this metric is described as 
        • Hit ratio at K (HR@K). It is equal to 1 if the test item appears
        in the list of k predicted items and 0 otherwise [13]. 
    
    But this is not what they implement, where they instead divide by k. 
    What they have actually implemented is more like Precision@k.
    However, Precision@k doesn't make a lot of sense in this context because
    there is only ONE possible correct answer in the list of generated 
    recommendations.  I don't think this is the best metric to use but 
    I'll keep it here for posterity. 

    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    hratk_score = 0
    for query_item, ground_truth in test:
        # If the query item and next item are the same, prediction is automatically correct
        if query_item == ground_truth:
            hratk_score += 1 / k
        else:
            # get the k most similar items to the query item (computes cosine similarity)
            neighbors = embeddings.similar_by_vector(query_item, topn=k)
            # clean up the list
            recommendations = [item for item, score in neighbors]
            # check if ground truth is in the recommedations
            if ground_truth in recommendations:
                hratk_score += 1 / k
    hratk_score /= len(test)
    return hratk_score*1000


def mrr_at_k(test, embeddings, k: int) -> float:
    """
    Mean Reciprocal Rank. 

    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    mrratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            # identify where the item is in the list
            rank_idx = (
                np.argwhere(np.array(recommendations) == ground_truth)[0][0] + 1
            )
            # score higher-ranked ground truth higher than lower-ranked ground truth
            mrratk_score += 1 / rank_idx
    mrratk_score /= len(test)
    return mrratk_score


def mrr_at_k_baseline(test, comatrix, k: int = 10) -> float:
    """
    Mean Reciprocal Rank. 

    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    mrratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        try:
            co_occ = collections.Counter(comatrix[query_item])
            items_and_counts = co_occ.most_common(k)
            recommendations = [item for (item, counts) in items_and_counts]
            if ground_truth in recommendations: 
                rank_idx = (
                    np.argwhere(np.array(recommendations) == ground_truth)[0][0] + 1
                )
                mrratk_score += 1 / rank_idx
        except:
            pass
    mrratk_score /= len(test)
    return mrratk_score