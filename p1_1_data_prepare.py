# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
"""
description: 对中国34个城市的地点数据进行处理，由csv文件转换成 arr数据
"""
import csv

import matplotlib.pyplot as plt
import numpy as np


def data_prepare():
    """
    对中国34个城市的地点数据进行处理，由csv文件转换成 arr数据
    :return:
    """
    data_path = "resource/china_cities.csv"
    data_save_path = "resource/china_cities.npy"
    data_list = []
    with open(data_path,encoding='gb2312') as f:
        data_csv = csv.reader(f)

        for row in data_csv:
            row_list = row[0].split(";")
            data_list.append([float(row_list[1]),float(row_list[2])])
            print(row)

    data_arr = np.array(data_list)
    np.save(data_save_path,data_arr)

def plot_scatter():
    """
    绘制34个城市对应的散点图
    """
    data_path = "resource/china_cities.npy"
    cities_data = np.load(data_path)

    plt.scatter(cities_data[:,0],cities_data[:,1],marker="*")
    plt.xlabel("x|longitude")
    plt.ylabel("y|latitude")
    plt.title("city scatter")
    plt.show()

if __name__ == "__main__":
    # data_prepare()
    plot_scatter()
    print("done!")