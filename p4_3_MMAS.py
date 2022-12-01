# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
"""
description: 使用 最大最小蚂蚁系统算法(MMAS)来求解 中国34个城市的TSP问题
"""
import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
class Ant(object):
    """
    蚂蚁个体类
    """
    def __init__(self, start=0,cityNumber = 34):
        """
        对蚂蚁个体进行初始化
        """
        self.start = start                   # 蚂蚁访问的第一个城市
        self.cityNumber = cityNumber
        self.table = []                       # 蚂蚁访问过的城市
        self.table.append(start)
        self.score = 0                        # 访问完所有城市之后，对应的适应度
        self.left_city_table = []             # 未访问的城市

    def cal_table(self):
        """
        根据 table 来确定left_city_table的值
        """
        for i in range(self.cityNumber):
            if i not in  self.table:
                self.left_city_table.append(i)

class ACO(object):
    """
    蚁群优化算法类
    """
    def __init__(self, antNumber=50,cities_path="resource/china_cities.npy"):
        self.antNumber = antNumber              # 蚂蚁数量
        self.alpha = 1                          # 信息素重要程度因子
        self.beta = 5                           # 启发函数重要因子
        self.rho = 0.1                          # 信息素挥发因子
        self.Q = 1                              # 常量系数
        ## 新增加的参数
        self.tau_min = -20
        self.tau_max = 20

        self.generation = 0
        # 初始化城市地点信息
        self.cities_data = np.load(cities_path)
        self.cityNumber = self.cities_data.shape[0]
        # 初始化相关矩阵
        self.tau = np.ones([self.cityNumber, self.cityNumber])*self.tau_max  # 信息素矩阵
        self.dis_map = self.gen_dis_map()       # 距离图矩阵
        self.eta = 1.0/self.dis_map             # 启发式矩阵

        self.ants = []                          # 蚁群集合
        self.best = None                        # 当前代数下，最优的个体

        # 初始化蚁群
        self.initialAnts()

    def gen_dis_map(self):
        """
        得到各个城市间的距离图
        """
        dis_mat = np.zeros((self.cityNumber, self.cityNumber))
        for i in range(self.cityNumber):
            for j in range(self.cityNumber):
                if i == j:
                    # 令对角线元素为无穷大
                    dis_mat[i][j] = np.inf
                    continue
                tmp = np.sqrt((self.cities_data[i,0] - self.cities_data[j,0]) ** 2 + (self.cities_data[i,1] - self.cities_data[j,1]) ** 2)
                dis_mat[i,j] = tmp
        return dis_mat

    def initialAnts(self):
        """
        初始化蚁群信息:为每一个蚂蚁选择第一个城市
        """
        self.ants = []
        for i in range(self.antNumber):
            ant_item = Ant(start=np.random.randint(self.cityNumber-1))
            ant_item.cal_table()
            self.ants.append(ant_item)

    def route_iteration(self):
        """
        一直计算，直到table填满
        """
        for i in range(self.antNumber):
            ant_item = self.ants[i]
            cur_city = ant_item.start
            table = ant_item.table
            left_city_table = ant_item.left_city_table
            # 一直计算，直到table填满
            # while len(table) < self.cityNumber:
            while len(left_city_table) != 0 :
                # 计算城市间的转移概率
                prob = []
                for j in left_city_table:
                    prob_item = self.tau[cur_city,j] ** self.alpha * self.eta[cur_city,j] ** self.beta
                    prob.append(prob_item)
                prob = np.array(prob)
                prob = prob/np.sum(prob)

                # 根据概率选择 下一个城市
                select_index = self.select_roulette(prob)
                cur_city = left_city_table[select_index]
                table.append(cur_city)
                left_city_table.remove(cur_city)

            # 更新ant
            self.ants[i].table = table
            self.ants[i].left_city_table = left_city_table
            self.ants[i].score = self.score_cal(table)
            if self.ants[i].score == 0:
                print(table)


        # 求出当前蚁群中的 最优蚂蚁
        self.best = self.ants[0]
        for ant_item in self.ants:
            if ant_item.score > self.best.score:
                self.best = ant_item


    def select_roulette(self,prob):
        """
        根据概率信息，通过轮盘赌的方式，选择一个序号
        """
        x = random.uniform(0, 1)
        for index in range(prob.shape[0]):
            prob_item = prob[index]
            x = x - prob_item
            if x <=0:
                return index


    def score_cal(self,gene_item):
        """
        计算适应度信息
        """
        distance = 0.0
        for i in range(-1, self.cityNumber-1):
            index1, index2 = gene_item[i-1], gene_item[i]
            distance += math.sqrt((self.cities_data[index1,0] - self.cities_data[index2,0]) ** 2 + (self.cities_data[index1,1] - self.cities_data[index2,1])** 2)

        return 1.0/distance

    def update_tau(self):
        """
        更新信息素
        """
        # 1.选择最优路径
        best_item = self.ants[0]
        for ant_item in self.ants:
            if ant_item.score > best_item.score:
                best_item = ant_item
        # 2.更新信息素
        best_item.score = self.score_cal(best_item.table)
        delta_tau = 1 / best_item.score

        # 3.根据 蚂蚁个体 走过的路径，来更新信息素
        for city_index in range(-1, self.cityNumber - 1):
            city0 = best_item.table[city_index]
            city1 = best_item.table[city_index + 1]
            # 计算蒸发后的 信息素
            self.tau[city0, city1] = (1 - self.rho) * self.tau[city0, city1] + self.rho * delta_tau

        # 4.最大最小范围约束
        for i in range(self.cityNumber):
            for j in range(self.cityNumber):
                if self.tau[i,j] > self.tau_max:
                    self.tau[i, j] = self.tau_max
                elif self.tau[i,j] < self.tau_min:
                    self.tau[i, j] = self.tau_min

    def aco_iteration(self):
        """
        进行一轮迭代
        """
        # 1.初始化蚁群
        self.initialAnts()
        # 2.蚁群跑完一圈
        self.route_iteration()

        # # 3.更新信息素
        self.update_tau()

        # 4.返回当前最优解
        self.generation =self.generation+ 1
        return self.best



def cal_tsp(iter_num):
    """
    对tsp问题进行迭代求解
    迭代次数为iter_num
    :return:
    """
    aco = ACO()
    distance_list = []
    global_best = Ant(start=0)
    while iter_num > 0:
        best_partical = aco.aco_iteration()
        if best_partical.score > global_best.score:
            global_best = best_partical

        distance = 1 / global_best.score
        if iter_num % 50 == 0:
            print(("%d : %f") % (aco.generation, distance))
        distance_list.append(distance)
        iter_num = iter_num - 1
    # 记录损失信息
    loss = np.array(distance_list)
    np.save("resource/aco_loss_3.npy", loss)

    # 输出相关结果
    print("route:", aco.best.table)
    fianl_route = aco.cities_data[aco.best.table]
    plt.subplot(2, 1, 1)
    plt.scatter(fianl_route[:, 0], fianl_route[:, 1])
    fianl_route = np.vstack([fianl_route, fianl_route[0]])
    plt.plot(fianl_route[:, 0], fianl_route[:, 1])
    plt.title('Route')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    iterations = np.arange(len(distance_list))
    best_record = distance_list
    plt.plot(iterations, best_record)
    plt.text(iterations[-1] - 20, distance_list[-1] + 10, '%.1f' % distance_list[-1], fontdict={'fontsize': 14})
    plt.title('Distance')
    plt.show()

if __name__ == "__main__":
    item_num = 500
    cal_tsp(item_num)