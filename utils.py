"""
Dataset preprocessing.

@date: 2022-3-12 10:22:21 update
"""
import pandas as pd
import numpy as np
import random

import sklearn
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from load_movie import *
import load_movie as lm


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def sparseFeature_demographic(feat, feat_num, demo_embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param demo_embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': demo_embed_dim}


def create_ml_1m_dataset(file, user_file, trans_score=2, embed_dim=8, demographic_emb_dim=8, test_neg_num=100, dataset=0, gamma = 0.5):
    """
    :param gamma: Scalar.
    :param demographic_emb_dim: test
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param test_neg_num: A scalar. The number of test negative samples
    :param user_file: A string. dataset path.
    :param dataset: A scalar. Select dataset.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    if dataset == 0:    # 1m
        data_df = pd.read_csv(file, sep="::", engine='python',      # 读取数据
                              names=['user_id', 'item_id', 'label', 'Timestamp'])
    elif dataset == 1:  # 100k
        data_df = pd.read_csv(file, sep=",", engine='python',  # 读取数据
                              names=['user_id', 'item_id', 'label'], skiprows=1)
    else:   # default
        data_df = pd.read_csv(file, sep="::", engine='python',  # 读取数据
                              names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    # data_df = data_df[data_df.item_count >= 5]      # 筛除观看次数小于5的电影
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = sklearn.utils.shuffle(data_df, random_state=0)    # seed = 0
    data_df = data_df.sort_values(by=['user_id'])  # 排序 user
    # data_df = data_df.sort_values(by=['user_id', 'Timestamp'])  # 排序 user, 观看次序


    # extra
    # movie_list, gender_num, age_num, occ_num, user_list = lm.get_movies_label(data_df, user_file, dataset)
    movie_list, gender_num, age_num, occ_num, user_list = lm.get_movies_label_bys(data_df, user_file, dataset, gamma)

    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id', 'label']].groupby('user_id')):
        # df = df[df['label'].isin(['4', '5'])]
        pos_list = df['item_id'].tolist()   # pos列表为交互过的所有项目

        def gen_neg():  # 随机一个未交互项目作为neg_item, 返回neg_item
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):   # pos_list的长度
            # hist_i = pos_list[:i]
            if i == len(pos_list) - 1:                  # 测试集: 1个正确pos + test_neg_num(arg) + 1
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])    # This is a list.
                temp_neg_gender_list = []
                temp_neg_age_list = []
                temp_neg_occ_list = []

                for neg in range(0, test_neg_num + 1):  # Creat new lists for each user's neg items.
                    temp_neg_movie = movie_list[neg_list[i + neg] - 1]
                    temp_neg_gender_list.append(temp_neg_movie.gender_label)
                    temp_neg_age_list.append(temp_neg_movie.age_label)
                    temp_neg_occ_list.append(temp_neg_movie.occ_label)
                test_data['neg_gender_label'].append(temp_neg_gender_list)
                test_data['neg_age_label'].append(temp_neg_age_list)
                test_data['neg_occ_label'].append(temp_neg_occ_list)

                temp_pos_movie = movie_list[pos_list[i] - 1]  # pos label
                test_data['pos_gender_label'].append(temp_pos_movie.gender_label)
                test_data['pos_age_label'].append(temp_pos_movie.age_label)
                test_data['pos_occ_label'].append(temp_pos_movie.occ_label)

                temp_user = user_list[user_id - 1]
                test_data['user_gender'].append(temp_user.gender)
                test_data['user_age'].append(temp_user.age)
                test_data['user_occ'].append(temp_user.occ)

            elif i == len(pos_list) - 2:                # 验证集长度: 1
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
                temp_pos_movie = movie_list[pos_list[i] - 1]  # pos label
                val_data['pos_gender_label'].append(temp_pos_movie.gender_label)
                val_data['pos_age_label'].append(temp_pos_movie.age_label)
                val_data['pos_occ_label'].append(temp_pos_movie.occ_label)
                temp_neg_movie = movie_list[neg_list[i] - 1]
                val_data['neg_gender_label'].append(temp_neg_movie.gender_label)
                val_data['neg_age_label'].append(temp_neg_movie.age_label)
                val_data['neg_occ_label'].append(temp_neg_movie.occ_label)
                temp_user = user_list[user_id - 1]
                val_data['user_gender'].append(temp_user.gender)
                val_data['user_age'].append(temp_user.age)
                val_data['user_occ'].append(temp_user.occ)
            else:
                train_data['user_id'].append(user_id)   # 训练集长度: len - 2
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
                temp_pos_movie = movie_list[pos_list[i] - 1]    # pos label
                train_data['pos_gender_label'].append(temp_pos_movie.gender_label)
                train_data['pos_age_label'].append(temp_pos_movie.age_label)
                train_data['pos_occ_label'].append(temp_pos_movie.occ_label)
                temp_neg_movie = movie_list[neg_list[i] - 1]
                train_data['neg_gender_label'].append(temp_neg_movie.gender_label)
                train_data['neg_age_label'].append(temp_neg_movie.age_label)
                train_data['neg_occ_label'].append(temp_neg_movie.occ_label)
                temp_user = user_list[user_id - 1]
                train_data['user_gender'].append(temp_user.gender)
                train_data['user_age'].append(temp_user.age)
                train_data['user_occ'].append(temp_user.occ)
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1     # 记录最大值
    item_feat_col = [sparseFeature('user_id', user_num, embed_dim),
                     sparseFeature('item_id', item_num, embed_dim),
                     sparseFeature_demographic('gender_label', gender_num, demographic_emb_dim),
                     sparseFeature_demographic('age_label', age_num, demographic_emb_dim),
                     sparseFeature_demographic('occ_label', occ_num, demographic_emb_dim)]
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']),
             np.array(train_data['neg_id']),
             np.array(train_data['pos_gender_label']), np.array(train_data['pos_age_label']),
             np.array(train_data['pos_occ_label']),
             np.array(train_data['neg_gender_label']), np.array(train_data['neg_age_label']),
             np.array(train_data['neg_occ_label']),
             np.array(train_data['user_gender']), np.array(train_data['user_age']),
             np.array(train_data['user_occ'])]

    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']),
           np.array(val_data['neg_id']),
           np.array(val_data['pos_gender_label']), np.array(val_data['pos_age_label']),
           np.array(val_data['pos_occ_label']),
           np.array(val_data['neg_gender_label']), np.array(val_data['neg_age_label']),
           np.array(val_data['neg_occ_label']),
           np.array(val_data['user_gender']), np.array(val_data['user_age']),
           np.array(val_data['user_occ'])]

    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']),
            np.array(test_data['neg_id']),
            np.array(test_data['pos_gender_label']), np.array(test_data['pos_age_label']),
            np.array(test_data['pos_occ_label']),
            np.array(test_data['neg_gender_label']), np.array(test_data['neg_age_label']),
            np.array(test_data['neg_occ_label']),
            np.array(test_data['user_gender']), np.array(test_data['user_age']),
            np.array(test_data['user_occ'])]
    print('============Data Preprocess End=============')
    return item_feat_col, train, val, test


