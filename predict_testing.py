#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: predict_testing.py
@time: 2018/10/19 14:27
@desc:
"""
from libsvm.python.svmutil import *
import numpy as np
import const
import os

prob_norm_testing_data = np.loadtxt("./data/norm_testing_data.data", dtype=np.float64, delimiter=" ").tolist()
self_norm_rand_data = np.loadtxt('./data/self_norm_rand_data.data', dtype=np.float64, delimiter=" ")
prob_x = self_norm_rand_data[:, 0:const.dim_num].tolist()
prob_y = self_norm_rand_data[:, const.dim_num].tolist()
"""Train Once"""
# cmd = '-t 2 -c ' + str(2 ** (- 0.4)) + ' -g ' + str(2 ** 1.4)
# model = svm_train(prob_y, prob_x, cmd)
# svm_save_model("./data/rand_model", model)

model = svm_load_model('./data/rand_model')
predict_label, accuracy, decision_value = svm_predict([0] * len(prob_norm_testing_data),
                                                      prob_norm_testing_data, model)
print(accuracy)
np.savetxt('./data/summit.csv', np.array(predict_label), fmt="%d", delimiter="\n")
print(predict_label.count(1))
print(predict_label.count(-1))