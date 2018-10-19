#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: contour.py.py
@time: 2018/10/19 18:50
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Z = np.loadtxt('./data/recg_mat.data', dtype=np.float)
c_scale = np.linspace(-5, 15, 11, dtype=np.int)
g_scale = np.linspace(-15, 3, 10, dtype=np.int)
G, C = np.meshgrid(g_scale,c_scale)
fig = plt.figure("3D")
ax = Axes3D(fig)
ax.plot_surface(G, C, Z,  rstride=2, cstride=2, cmap=plt.get_cmap('rainbow'))
plt.xlabel("G")
plt.ylabel("C")
ax.set_zlabel("Accuracy")
ax.set_zlim(bottom=70, top=76)
ax.set_xticks(np.arange(-15, 3, 2))
ax.set_yticks(np.arange(-5, 15, 2))

plt.figure("contour")
plt.contourf(G, C, Z, 20, alpha = 0.6, cmap = plt.get_cmap('rainbow'))
C = plt.contour(G, C, Z, 20, colors = 'black')
plt.xticks(np.arange(-15,5,2))
plt.yticks(np.arange(-5,17,2))
plt.show()