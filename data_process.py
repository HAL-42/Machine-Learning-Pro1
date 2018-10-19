#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: data_process.py
@time: 2018/10/14 15:06
@desc: Process the data
"""
import numpy as np
import const
import json

raw_data = np.loadtxt('./data/training.data', dtype=np.float, delimiter=" ")
"""Calculate the proportion of each class in the data set"""
y = raw_data[:,6].tolist()
# print(y)
# print(y.count(-1)/len(y))
"""Shuffle the raw data
ATTENTION: DO IT ONLY ONCE 
"""
# np.random.shuffle(raw_data)
# np.savetxt('./data/rand_training.data', raw_data, fmt='%.5f', delimiter=" ")
"""Get data set for cross validation
ATTENTION: DO IT ONLY ONCE 
"""
# rand_data = np.loadtxt('./data/rand_training.data', dtype=np.float, delimiter=" ")
# crs_vld_data = rand_data[-const.crs_vld_data_num-1-10000:-1-10000,:]     # Choose data from the middle of the data set
# y = crs_vld_data[:,6].tolist()
# print(y.count(-1)/len(y))                                                # 0.499
# np.savetxt('./data/crs_vld_data.data', crs_vld_data, fmt='%.5f', delimiter=" ")
"""Normalization"""
# crs_vld_data = np.loadtxt('./data/crs_vld_data.data', dtype=np.float, delimiter=" ")
# mean = [np.mean(crs_vld_data[:, i]) for i in range(const.dim_num)]
# std = [np.std(crs_vld_data[:, i]) for i in range(const.dim_num)]
# print(mean)
# print(std)
# with open('./data/crs_vld_data_mean.json', 'w') as f:
#     json.dump(mean, f)
# with open('./data/crs_vld_data_std.json', 'w') as f:
#     json.dump(std, f)
#
# np_mean = np.array(mean)
# np_std = np.array(std)
# for row in crs_vld_data:
#     row[0:const.dim_num] = (row[0:const.dim_num] - np_mean) / np_std
# rand_data = np.loadtxt('./data/rand_training.data', dtype=np.float, delimiter=" ")
# for row in rand_data:
#     row[0:const.dim_num] = (row[0:const.dim_num] - np_mean) / np_std
#
# _mean = [np.mean(crs_vld_data[:, i]) for i in range(const.dim_num)]         # Validate the accuracy of normalization
# _std = [np.std(crs_vld_data[:, i]) for i in range(const.dim_num)]
# _rand_mean = [np.mean(rand_data[:, i]) for i in range(const.dim_num)]
# _rand_std = [np.std(rand_data[:, i]) for i in range(const.dim_num)]
# print(_mean)
# print(_std)
# print(_rand_mean)
# print(_rand_std)
#
# np.savetxt('./data/norm_crs_vld_data.data', crs_vld_data, fmt='%.18e', delimiter=" ")
# np.savetxt('./data/norm_rand_data.data', rand_data, fmt='%.18e', delimiter=" ")
# np.savetxt('./data/norm_crs_vld_x.data',crs_vld_data[:, 0:const.dim_num], fmt='%.18e', delimiter=" ")
# np.savetxt('./data/norm_crs_vld_y.data',crs_vld_data[:, const.dim_num], fmt='%.18e', delimiter=" ")
# np.savetxt('./data/norm_rand_x.data', rand_data[:, 0:const.dim_num], fmt='%.18e', delimiter=" ")
# np.savetxt('./data/norm_rand_y.data', rand_data[:, const.dim_num], fmt='%.18e', delimiter=" ")













