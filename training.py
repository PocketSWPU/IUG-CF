"""
Train CFIUG model.

@date:2022-3-4 14:04:58
"""
import os
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.optimizers import Adam
import numpy as np

from evaluate import *
from IUGCF import IUGCF
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1:all info; 2:warning and Error; 3:Error

if __name__ == '__main__':

    # ========================= Datasets ==============================
    dataset = 0  # 0:1m; 1:100k

    if dataset == 0:
        file = 'datasets/movielens-1M/ratings.dat'
        user_file = 'datasets/movielens-1M/users.dat'
    elif dataset == 1:
        file = 'datasets/movielens-100K/rate.CSV'
        user_file = 'datasets/movielens-100K/user.CSV'
    else:
        file = 'datasets/movielens-1M/ratings.dat'
        user_file = 'datasets/movielens-1M/users.dat'

    # ========================= Hyper Parameters =======================
    trans_score = 1
    test_neg_num = 100

    demographic_emb_dim = 128
    embed_dim = 512  # 64 default
    hidden_units = [512]  # [512, 256]
    embed_reg = 1e-6  # 1e-6
    activation = 'relu'
    dropout = 0.2
    K = 10
    gamma = 0.2  #
    reg_ord = 1  #
    reg_coe = 0.001  #

    learning_rate = 0.001
    epochs = 40
    batch_size = 256  # 512 default

    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_ml_1m_dataset(file, user_file, trans_score, embed_dim,
                                                             demographic_emb_dim, test_neg_num, dataset, gamma)

    # ============================Build Model==========================
    model = IUGCF(feature_columns, hidden_units, dropout, activation, embed_reg, reg_ord, reg_coe)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    best_result = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        #     t1 = time()
        model.fit(
            train,
            None,
            validation_data=(val, None),
            epochs=1,
            batch_size=batch_size,
        )
        # ===========================Test==============================
        #     t2 = time()
        # if epoch % 2 == 0:
        hr_10, ndcg_10, hr_5, ndcg_5 = evaluate_model_new(model, test)
        print('Iteration %d : HR@5 = %.4f, NDCG@5 = %.4f, HR@10 = %.4f, NDCG@10 = %.4f'
              % (epoch, hr_5, ndcg_5, hr_10, ndcg_10))
        results.append([epoch, hr_5, ndcg_5, hr_10, ndcg_10])

        best_result.append([hr_5, ndcg_5, hr_10, ndcg_10])
    # ========================== Write Log ===========================
    # savetime = time.strftime("%m-%d-%H-%M", time.localtime())
    best_result = np.array(best_result).max(axis=0)
    dt = time.strftime('%m%d-%H%M', time.localtime(time.time()))    # date-time

    # ========================== Best ================================
    print('Best performance : HR@5 = %.4f, NDCG@5 = %.4f, HR@10 = %.4f, NDCG@10 = %.4f'
          % (best_result[0], best_result[1], best_result[2], best_result[3]))
    results.append(['Best', best_result[0], best_result[1], best_result[2], best_result[3]])

    # ========================== Save ================================
    pd.DataFrame(results, columns=['Iteration', 'hr_10', 'ndcg_10', 'hr_5', 'ndcg_5']) \
        .to_csv('log/{}/IUG-CF_log_dim_{}+{}__dataset_{}_{}.csv' \
                .format(dataset, embed_dim, demographic_emb_dim, dataset, dt),
                index=False)
