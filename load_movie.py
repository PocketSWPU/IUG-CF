"""
Load items.

@date : 2022-3-12 10:24:52 update.
"""

import numpy as np
from utils import *
import pandas as pd


class User:

    def __init__(self, index, gender, age, occ, dict_gender, dict_age):
        self.index = int(index)
        self.gender = dict_gender.get(gender)
        self.age = dict_age.get(age)
        self.occ = occ


class Movie:

    def __init__(self, gender_label, age_label, occ_label):
        self.gender_label = gender_label
        self.age_label = age_label
        self.occ_label = occ_label


def load_user_info(user_file, dataset):
    """
    :param user_file: user.dat
    :param dataset: scalar.
    :return: list: all user.
    """
    if dataset == 0:
        data_df = pd.read_csv(user_file, sep="::", engine='python',  # 读取数据
                              names=['user_id', 'gender', 'age', 'occ', 'place'])
        data_df.drop(['place'], axis=1, inplace=True)  # 删除多余列
    elif dataset == 1:
        data_df = pd.read_csv(user_file, sep=",", engine='python',  # 读取数据
                              names=['user_id', 'gender', 'age', 'occ', 'place'], skiprows=1)
        data_df.drop(['place'], axis=1, inplace=True)  # 删除多余列
    else:
        data_df = pd.read_csv(user_file, sep="::", engine='python',  # 读取数据
                              names=['user_id', 'gender', 'age', 'occ', 'place'])

    user_list = []  # 储存用户信息
    dict_gender = {'M': 0, 'F': 1}  # 性别字典
    dict_age = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}  # 年龄字典
    for row in data_df.itertuples():  # 初始化
        user_id = row[1]
        gender = row[2]
        age = row[3]
        occ = row[4]
        user_list.append((User(user_id, gender, age, occ, dict_gender, dict_age)))
    user_num = int(data_df['user_id'].max())
    gender_num = len(dict_gender)
    age_num = len(dict_age)
    occ_num = int(data_df['occ'].max()) + 1
    return user_list, user_num, gender_num, age_num, occ_num


def get_movies_label(rate_data_df, user_file, dataset):
    """
    拉普拉斯平滑
    :param rate_data_df: Dataframe.
    :param user_file: String.
    :param dataset: Scalar.
    :return:movie_list, gender_num, age_num, occ_num, user_list
    """
    user_list, user_num, gender_num, age_num, occ_num = load_user_info(user_file, dataset)  # 用户list 下标+1 = user_id
    movie_list = []
    movie_num = rate_data_df['item_id'].max()
    # 用三个数组分别存储人口信息
    # 注: 这里np.ones : 默认分子加一，是拉普拉斯平滑
    gender_rate = np.ones((movie_num, gender_num))
    age_rate = np.ones((movie_num, age_num))
    occ_rate = np.ones((movie_num, occ_num))

    # 记录次数
    # 注: 这里后面的 *n 表示分母加上总数，是拉普拉斯平滑
    gender_count = np.ones((movie_num, gender_num)) * 2
    age_count = np.ones((movie_num, age_num)) * 7
    occ_count = np.ones((movie_num, occ_num)) * 21

    # 遍历rate_df 统计数据
    for row in rate_data_df.itertuples():
        user_id, item_id, rate = row[1] - 1, row[2] - 1, row[3]
        temp_gender = user_list[user_id].gender  # 当前数据
        temp_age = user_list[user_id].age
        temp_occ = user_list[user_id].occ

        gender_rate[item_id][temp_gender] += rate  # 统计
        age_rate[item_id][temp_age] += rate
        occ_rate[item_id][temp_occ] += rate

        gender_count[item_id][temp_gender] += 1  # 计数
        age_count[item_id][temp_age] += 1
        occ_count[item_id][temp_occ] += 1

    # for i in gender_rate:
    #     for i_2 in i:
    #         gender_rate[i][i_2] = (gender_rate[i][i_2] + 1) / (gender_count[i][i_2] + 2)
    # for j in age_rate:
    #     for j_2 in j:
    #         age_rate[j][j_2] = (age_rate[j][j_2] + 1) / (gender_count[j][j_2] + 7)
    # for k in occ_rate:
    #     for k_2 in k:
    #         occ_rate[k][k_2] = (gender_rate[k][k_2] + 1) / (gender_count[k][k_2] + 21)

    # gender_rate /= gender_count
    # age_rate /= age_count
    # occ_rate /= occ_count

    for temp_movie_index in range(movie_num):
        gender_label = np.argmax(gender_rate[temp_movie_index])
        age_label = np.argmax(age_rate[temp_movie_index])
        occ_label = np.argmax(occ_rate[temp_movie_index])
        temp_movie = Movie(gender_label, age_label, occ_label)
        movie_list.append(temp_movie)

    return movie_list, gender_num, age_num, occ_num, user_list


def get_movies_label_bys(rate_data_df, user_file, dataset, gamma):
    """
    :param rate_data_df:
    :param user_file:
    :param dataset:
    :param gamma: Scalar. 缩放因子.
    :return:
    """
    user_list, user_num, gender_num, age_num, occ_num = load_user_info(user_file, dataset)  # 用户list 下标+1 = user_id
    movie_list = []
    movie_num = rate_data_df['item_id'].max()
    # rate数组记录评分
    gender_rate = np.zeros((movie_num, gender_num))
    age_rate = np.zeros((movie_num, age_num))
    occ_rate = np.zeros((movie_num, occ_num))

    # count数组记录次数
    gender_count = np.zeros((movie_num, gender_num))
    age_count = np.zeros((movie_num, age_num))
    occ_count = np.zeros((movie_num, occ_num))

    # 遍历rate_df 统计数据
    for row in rate_data_df.itertuples():
        user_id, item_id, rate = row[1] - 1, row[2] - 1, row[3]
        temp_gender = user_list[user_id].gender  # 当前数据
        temp_age = user_list[user_id].age
        temp_occ = user_list[user_id].occ

        gender_rate[item_id][temp_gender] += rate  # 统计评分
        age_rate[item_id][temp_age] += rate
        occ_rate[item_id][temp_occ] += rate

        gender_count[item_id][temp_gender] += 1  # 计数
        age_count[item_id][temp_age] += 1
        occ_count[item_id][temp_occ] += 1
    # R = count / num; m = rates /
    for temp_movie_index in range(movie_num):
        # new 分子
        gender_rate[temp_movie_index] += (np.sum(gender_rate[temp_movie_index]) / gender_num) * gamma
        age_rate[temp_movie_index] += (np.sum(age_rate[temp_movie_index]) / age_num) * gamma
        occ_rate[temp_movie_index] += (np.sum(occ_rate[temp_movie_index]) / occ_num) * gamma

        # new 分母
        gender_count[temp_movie_index] += (np.sum(gender_count[temp_movie_index]) / gender_num) * gamma
        age_count[temp_movie_index] += (np.sum(age_count[temp_movie_index]) / age_num) * gamma
        occ_count[temp_movie_index] += (np.sum(occ_count[temp_movie_index]) / occ_num) * gamma

    gender_final = (gender_rate + 1) / (gender_count + gender_num)
    age_final = (age_rate + 1) / (age_count + age_num)
    occ_final = (occ_rate + 1) / (occ_count + occ_num)

    for temp_movie_index in range(movie_num):
        gender_label = np.argmax(gender_final[temp_movie_index])
        age_label = np.argmax(age_final[temp_movie_index])
        occ_label = np.argmax(occ_final[temp_movie_index])
        temp_movie = Movie(gender_label, age_label, occ_label)
        movie_list.append(temp_movie)

    return movie_list, gender_num, age_num, occ_num, user_list
