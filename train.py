#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: train.py
@time: 2018/10/14 17:51
@desc: Training by SVM
"""
import numpy as np
import multiprocessing
import const
import libsvm.python.svm as svm
from libsvm.python.svmutil import *
import logging
import os
logging.basicConfig(filename='./data/log.txt', level=logging.DEBUG, format=const.LOG_FORMAT)


def solve_prob(y, x, parm_q, rslt_q):
    while not parm_q.empty():
        parm = parm_q.get()
        msg = " 这里是进程: %sd  父进程ID：%s c = %f g = %f" % (os.getpid(), os.getppid(), parm["c"], parm["g"])
        print(msg)
        logging.warning(msg)

        """First Solve"""
        # cmd = '-t 2 -c ' + str(2 ** parm["c"]) + ' -g ' + str(2 ** parm["g"]) + ' -v 5'
        """Second Solve"""
        # del_c = ((parm["g"] + 5.5) / 2) * 0.3
        # del_g = ((parm["c"] - 5.5) / 2) * 0.3
        """Third Solve"""
        del_c = 0
        del_g = 0
        """Forth Solve"""
        del_c = 0
        del_g = 0

        cmd = '-t 2 -c ' + str(2 ** (parm["c"] + del_c)) + ' -g ' + str(2 ** (parm["g"] + del_g)) + ' -v 5'
        recg = svm_train(y[:len(y)], x[:len(x)], cmd)
        rslt_q.put((parm, recg))
    msg = "进程: %sd 结束"% (os.getpid())
    print(msg)
    logging.warning(msg)


if __name__ == "__main__":
    """Transfer x to required form"""
    norm_crs_vld_x = np.loadtxt('./data/norm_crs_vld_x.data', dtype=np.float64, delimiter=" ")
    norm_rand_x = np.loadtxt('./data/norm_rand_x.data', dtype=np.float64, delimiter=" ")
    crs_vld_y = np.loadtxt('./data/norm_crs_vld_y.data', dtype=np.float64, delimiter=" ")
    rand_y = np.loadtxt('./data/norm_rand_y.data', dtype=np.float64, delimiter=" ")

    prob_crs_vld_x = []
    for row in norm_crs_vld_x:
        l_row = row.tolist()
        l_row = dict(enumerate(l_row))
        prob_crs_vld_x.append(l_row)

    prob_rand_x = []
    for row in norm_rand_x:
        l_row = row.tolist()
        l_row = dict(enumerate(l_row))
        prob_rand_x.append(l_row)

    prob_crs_vld_y = crs_vld_y.tolist()
    prob_rand_y = rand_y.tolist()

    crs_vld_prob = svm.svm_problem(prob_crs_vld_y, prob_crs_vld_x)

    """First Solve"""
    # c_scale = np.linspace(-5, 15, 11, dtype=np.int).tolist()
    # g_scale = np.linspace(-15, 3, 10, dtype=np.int).tolist()
    """Second Solve"""
    # c_scale = np.linspace(5.5, 25.5, 11, dtype=np.float).tolist()
    # g_scale = np.linspace(-5.5, 12.5, 10, dtype=np.float).tolist()
    """Third Solve"""
    # c_scale = np.linspace(7.5, 25.5, 10, dtype=np.float).tolist()
    # g_scale = np.linspace(-9.5, -1.5, 5, dtype=np.float).tolist()
    """Forth Solve"""
    c_scale = np.linspace(-2.8, 0.8, 19, dtype=np.float).tolist()
    g_scale = np.linspace(-0.8, 2.8, 19, dtype=np.float).tolist()

    manager = multiprocessing.Manager()
    parm_q = manager.Queue()
    rslt_q = manager.Queue()
    for c in c_scale:
        for g in g_scale:
            parm_q.put({"c": c, "g": g})

    pro = []
    for i in range(4):
        pro.append(multiprocessing.Process(target=solve_prob, args=(prob_crs_vld_y, prob_crs_vld_x, parm_q, rslt_q,),
                                           name="pro"+str(i)))
        pro[i].start()
    #p = multiprocessing.Pool(8)
    # for i in range(8):
    #     p.apply_async(svm_predict, args=(prob_crs_vld_y, prob_crs_vld_x, parm_q, rslt_q,))
    print("Solving...")
    # p.close()
    # p.join()
    for i in range(4):
        pro[i].join()
        msg = "主进程中进程pro%d结束" % (i)
        print("主进程中进程pro%d结束" % (i))
        logging.warning(msg)

    recg_mat = np.zeros((len(c_scale), len(g_scale)))

    while not rslt_q.empty():
        rslt = rslt_q.get()
        """First to Third Solve"""
        # recg_mat[int((rslt[0]["c"] - c_scale[0]) // 2)][int((rslt[0]["g"] - g_scale[0]) // 2)] = rslt[1]
        """Forth Solve"""
        recg_mat[int(np.round((rslt[0]["c"] - c_scale[0]) / 0.2))][int(np.round((rslt[0]["g"] - g_scale[0]) / 0.2))] = rslt[1]
    print(recg_mat)

    """First Solve"""
    # np.savetxt('./data/recg_mat.data', recg_mat, fmt="%18e", delimiter=" ")
    """Second Solve"""
    # np.savetxt('./data/recg_mat_second.data', recg_mat, fmt="%18e", delimiter=" ")
    """Third Solve"""
    np.savetxt('./data/recg_mat_Third.data', recg_mat, fmt="%18e", delimiter=" ")
    """Forth Solve"""
    np.savetxt('./data/recg_mat_Forth.data', recg_mat, fmt="%18e", delimiter=" ")

    os.system("pause")










