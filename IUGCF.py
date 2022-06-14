"""
IUG-CF model.

@date: 2022-3-12 10:24:01 update
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input

from modules import *


class IUGCF(Model):
    # define your layers in __init__()
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, reg_ord=1,
                 reg_cog=0.001, **kwargs):
        super(IUGCF, self).__init__(**kwargs)

        # feature columns
        self.user_fea_col, self.item_fea_col, self.gender_fea_col, self.age_fea_col, self.occ_fea_col = feature_columns
        emb_ini = tf.keras.initializers.RandomNormal(seed=2022)
        self.reg_ord = reg_ord
        self.reg_cog = reg_cog

        # MLP user embedding
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer=emb_ini,
                                            embeddings_regularizer=l2(embed_reg))
        # MLP item embedding
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer=emb_ini,
                                            embeddings_regularizer=l2(embed_reg))
        # gender
        self.gender_embedding = Embedding(input_dim=self.gender_fea_col['feat_num'],
                                          input_length=1,
                                          output_dim=self.gender_fea_col['embed_dim'],
                                          embeddings_initializer=emb_ini,
                                          embeddings_regularizer=l2(embed_reg))
        # age
        self.age_embedding = Embedding(input_dim=self.age_fea_col['feat_num'],
                                       input_length=1,
                                       output_dim=self.age_fea_col['embed_dim'],
                                       embeddings_initializer=emb_ini,
                                       embeddings_regularizer=l2(embed_reg))
        # occ
        self.occ_embedding = Embedding(input_dim=self.occ_fea_col['feat_num'],
                                       input_length=1,
                                       output_dim=self.occ_fea_col['embed_dim'],
                                       embeddings_initializer=emb_ini,
                                       embeddings_regularizer=l2(embed_reg))

        self.dnn_demographic = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        # self.dnn_mlp = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    #  you should implement the model's forward pass in call().
    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs, pos_gender_inputs, pos_age_inputs, pos_occ_inputs, neg_gender_inputs, \
        neg_age_inputs, neg_occ_inputs, user_gender, user_age, user_occ = inputs

        # - 初始化 - #
        user_embed = self.mlp_user_embedding(user_inputs)  # (None, 1, dim)
        pos_embed = self.mlp_item_embedding(pos_inputs)  # (None, 1, dim)
        neg_embed = self.mlp_item_embedding(neg_inputs)  # (None, 1/101, dim)

        pos_gender_embed = self.gender_embedding(pos_gender_inputs)  # (None, 1, dim)
        pos_age_embed = self.age_embedding(pos_age_inputs)  # (None, 1, dim)
        pos_occ_embed = self.occ_embedding(pos_occ_inputs)  # (None, 1, dim)
        neg_gender_embed = self.gender_embedding(neg_gender_inputs)  # (None, 1/101, dim)
        neg_age_embed = self.age_embedding(neg_age_inputs)  # (None, 1/101, dim)
        neg_occ_embed = self.occ_embedding(neg_occ_inputs)  # (None, 1/101, dim)
        user_gender_embed = self.gender_embedding(user_gender)  # (None, 1, dim)
        user_age_embed = self.age_embedding(user_age)  # (None, 1, dim)
        user_occ_embed = self.occ_embedding(user_occ)  # (None, 1, dim)

        # - 人口信息内积 - #
        pos_inner_age = tf.nn.sigmoid(tf.multiply(pos_age_embed, user_age_embed))
        pos_inner_gender = tf.nn.sigmoid(tf.multiply(pos_gender_embed, user_gender_embed))
        pos_inner_occ = tf.nn.sigmoid(tf.multiply(pos_occ_embed, user_occ_embed))

        neg_inner_age = tf.nn.sigmoid(tf.multiply(neg_age_embed, user_age_embed))
        neg_inner_gender = tf.nn.sigmoid(tf.multiply(neg_gender_embed, user_gender_embed))
        neg_inner_occ = tf.nn.sigmoid(tf.multiply(neg_occ_embed, user_occ_embed))

        # - 人口信息拼接 - #
        # pos_user_inner = tf.concat([pos_inner_age, pos_inner_gender, pos_inner_occ], axis=-1)   # 3*dm_dim
        # neg_user_inner = tf.concat([neg_inner_age, neg_inner_gender, neg_inner_occ], axis=-1)

        pos_user_inner = tf.add(tf.add(pos_inner_age, pos_inner_gender), pos_inner_occ)  # demo_dim * 1
        neg_user_inner = tf.add(tf.add(neg_inner_age, neg_inner_gender), neg_inner_occ)

        # - 人口信息 + 用户 + 项目 拼接 - #
        pos_vector = tf.concat([user_embed, pos_embed, pos_user_inner], axis=-1)  # 2*dim + demo_dim
        neg_vector = tf.concat([tf.tile(user_embed, multiples=[1, neg_embed.shape[1], 1]), neg_embed, neg_user_inner],
                               axis=-1)

        # DNN
        dnn_pos_vector = self.dnn_demographic(pos_vector)
        dnn_neg_vector = self.dnn_demographic(neg_vector)

        # den_pos = tf.concat([dnn_pos_vector, pos_mf], axis=-1)
        # den_neg = tf.concat([dnn_neg_vector, neg_mf], axis=-1)
        den_pos = dnn_pos_vector
        den_neg = dnn_neg_vector

        # Dense
        score_pos = self.dense(den_pos)
        score_neg = self.dense(den_neg)

        # squeeze
        pos = tf.squeeze(score_pos, axis=-1)
        neg = tf.squeeze(score_neg, axis=-1)

        # minus_pos : 差
        minus_pos = tf.subtract(tf.concat([pos_inner_age, pos_inner_gender, pos_inner_occ], axis=-1),
                                tf.concat([user_age_embed, user_gender_embed, user_occ_embed], axis=-1))
        minus_neg = tf.subtract(tf.concat([neg_inner_age, neg_inner_gender, neg_inner_occ], axis=-1),
                                tf.concat([user_age_embed, user_gender_embed, user_occ_embed], axis=-1))
        lam = tf.convert_to_tensor(self.reg_cog, dtype=tf.float32)       # 系数 \lambda

        l_2_pos = tf.linalg.norm(minus_pos, ord=self.reg_ord)
        l_2_neg = tf.linalg.norm(minus_neg, ord=self.reg_ord)

        # 1.minus_pos 求平均 + 绝对值
        ab_pos = tf.reduce_mean(tf.abs(minus_pos))
        ab_neg = tf.reduce_mean(tf.abs(minus_neg))
        # 2.处理
        lamb_pos = tf.math.reciprocal(tf.exp(ab_pos))
        lamb_neg = tf.math.reciprocal(tf.exp(ab_neg))
        # 3.pos
        pos = tf.multiply(pos, lamb_pos)
        neg = tf.multiply(neg, lamb_neg)
        # Loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg))) / 2 \
                 + tf.multiply(l_2_pos, lam) - tf.multiply(l_2_neg, lam)
        self.add_loss(losses)

        logits = tf.concat([pos, neg], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)  # [1,H]
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_gender_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_age_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_occ_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_gender_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_age_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_occ_inputs = Input(shape=(1,), dtype=tf.int32)
        user_gender = Input(shape=(1,), dtype=tf.int32)
        user_age = Input(shape=(1,), dtype=tf.int32)
        user_occ = Input(shape=(1,), dtype=tf.int32)

        Model(inputs=[user_inputs, pos_inputs, neg_inputs, pos_gender_inputs, pos_age_inputs, pos_occ_inputs,
                      neg_gender_inputs, neg_age_inputs, neg_occ_inputs, user_gender, user_age, user_occ],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs, pos_gender_inputs, pos_age_inputs, pos_occ_inputs,
                                 neg_gender_inputs, neg_age_inputs, neg_occ_inputs, user_gender, user_age,
                                 user_occ])).summary()
