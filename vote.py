#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: vote.py
@time: 2018/10/19 17:19
@desc:
"""
from libsvm.python.svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import const
import os

prob_crs_vld_y = np.loadtxt('./data/norm_crs_vld_y.data', dtype=np.float64, delimiter=" ").tolist()
prob_crs_vld_x = np.loadtxt('./data/norm_crs_vld_x.data', dtype=np.float64, delimiter=" ").tolist()
prob_rand_y = np.loadtxt('./data/norm_rand_y.data', dtype=np.float64, delimiter=" ").tolist()
prob_rand_x = np.loadtxt('./data/norm_rand_x.data', dtype=np.float64, delimiter=" ").tolist()

model=[]
# cmd = '-t 2 -n ' + str(1) + ' -g ' + str(4)
# model.append(svm_train(prob_crs_vld_y, prob_crs_vld_x, cmd))

cmd = '-t 2 -n ' + str(2 ** (- 0.4)) + ' -g ' + str(2 ** 1.4)
model.append(svm_train(prob_crs_vld_y, prob_crs_vld_x, cmd))

# cmd = '-t 2 -n ' + str(1) + ' -g ' + str(5.28)
# model.append(svm_train(prob_crs_vld_y, prob_crs_vld_x, cmd))

cmd = '-t 2 -n ' + str(1.1487) + ' -g ' + str(3.482)
model.append(svm_train(prob_crs_vld_y, prob_crs_vld_x, cmd))

cmd = '-t 2 -n ' + str(1) + ' -g ' + str(4)
model.append(svm_train(prob_crs_vld_y, prob_crs_vld_x, cmd))

label = np.zeros(10000,dtype=int)
for m in model:
    predict_label, accuracy, decision_value = svm_predict(prob_rand_y[0:10000], prob_rand_x[0:10000], m)
    label = label + np.array(predict_label)

_ = 0
for i in range(10000):
    if label[i] > 0:
        label[i] = 1
    else:
        label[i] = -1
    if label[i] == int(prob_rand_y[i]):
        _ += 1

print(label)
print(_)
print("Vote Accuracy:", _ / 10000)
os._exit(0)

label = list(map(int, prob_rand_y))
decision_value = list(map(lambda x: x[0], decision_value))
tmp = []
for i in range(len(decision_value)):
    tmp.append([decision_value[i], label[i]])
tmp.sort(key=lambda x: x[0], reverse=True)

sorted_decision_value = []
sorted_label = []
for x, y in tmp:
    sorted_decision_value.append(x)
    sorted_label.append(y)

print(sorted_label)
tpr = [0] * (len(sorted_decision_value))
fpr = [0] * (len(sorted_decision_value))
p_num = sorted_label.count(1)
n_num = sorted_label.count(-1)

tp = 0
fp = 0
min_delta = const.INFINIT
index = -1
for i in range(len(sorted_decision_value)):
    if sorted_label[i] == 1:
        tp += 1
    else:
        fp += 1
    tpr[i] = tp / p_num
    fpr[i] = fp / n_num
    if abs(tpr[i] + fpr[i] -1) < min_delta:
        min_delta = abs(tpr[i] + fpr[i] -1)
        index = i
tpr = [0] + tpr
fpr = [0] + fpr

AUC = 0
for i in range(1, len(sorted_decision_value)):
    AUC += (fpr[i] - fpr[i - 1]) * tpr[i]
print(AUC)

print(len(tpr))
plt.figure("ROC Curve")
plt.plot(fpr, tpr, color="red", linewidth=1)
plt.plot(np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), color="blue", linewidth=0.5)
plt.annotate("Intersection fpr={0:.5f}, tpr={1:.5f}".format(fpr[index], tpr[index]), xy=(fpr[index], tpr[index]),
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.show()
