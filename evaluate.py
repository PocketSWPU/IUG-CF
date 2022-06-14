"""
Evaluate IUG-CF model.
Metrics:hr, ndcg, mrr.

@date:2022-3-4 14:12:23
"""
import numpy as np


def evaluate_model(model, test, k):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param k: top k
    :return: hit rate, ndcg, mrr
    """
    pred_y = - model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg, mrr = 0.0, 0.0, 0.0
    count = len(rank)
    for r in rank:
        if r < k:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
            mrr += 1 / (r + 1)
    return hr / count, ndcg / count, mrr / count


def evaluate_model_new(model, test):
    """
    evaluate model
    :param model: model
    :param test: test set
    :return: hr,ndcg @ 10, 5
    """
    pred_y = - model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr_10, ndcg_10, hr_5, ndcg_5 = 0.0, 0.0, 0.0, 0.0
    count = len(rank)
    for r in rank:
        if r < 10:
            hr_10 += 1
            ndcg_10 += 1 / np.log2(r + 2)
            if r < 5:
                hr_5 += 1
                ndcg_5 += 1 / np.log2(r + 2)
    return hr_10 / count, ndcg_10 / count, hr_5 / count, ndcg_5 / count
