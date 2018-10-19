#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: const.py
@time: 2018/10/14 15:08
@desc: Add const to project
"""


class _const(object):

    class ConstError(PermissionError):pass

    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name]=value

        def __delattr__(self, name):
            if name in self.__dict__:
                raise self.ConstError("Can't unbind const(%s)" % name)
            raise NameError(name)


import sys
sys.modules[__name__]=_const()

_const.dim_num = 6
_const.raw_data_num = 44610
_const.rand_data_num = _const.raw_data_num
_const.crs_vld_data_num = 43000
_const.LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_const.INFINIT = 1e8
