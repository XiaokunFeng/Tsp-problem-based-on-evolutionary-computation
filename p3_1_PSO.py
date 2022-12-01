# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
"""
description: 使用基本的粒子群优化算法来求解 中国34个城市的TSP问题
"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
class Partical(object):
    """
    粒子类
    """
    def __init__(self, gene=None):
        """
        对粒子进行初始化，通过给定gene,或gene_encode的方式
        """
        self.cur_gene = gene                        # 粒子当前的基因信息
        self.cur_score = 0                          # 粒子当前的适应度
        self.best_gene = gene                       # 粒子最优的基因信息
        self.best_score = 0                         # 粒子最优的适应度

    def update(self):
        """
        在粒子的cur_gene，cur_score信息更新之后，更新 粒子最优信息
        """
        if self.best_score <self.cur_score:
            self.best_score = self.cur_score
            self.best_gene = self.cur_gene

class PSO(object):
    """
    粒子群优化算法类
    """
    def __init__(self, particalCount=100,cities_path="resource/china_cities.npy"):
        self.particalCount = particalCount                      # 粒子数量                            # 交叉率

        self.particals = []                                     # 各个粒子信息的集合
        self.best = None                                        # 当前代数下，最优的个体

        self.generation = 0
        # 初始化城市地点信息
        self.cities_data = np.load(cities_path)
        self.cityNumber = self.cities_data.shape[0]
        
        # 初始化粒子群数据
        self.initialGeneration()

    def initialGeneration(self):
        """
        初始化粒子群信息
        """
        self.particals = []
        # 为种群中的各个个体赋值
        for _ in range(self.particalCount):
            gene = []
            for gene_item in range(self.cityNumber):
                gene.append(gene_item)
            # 乱序处理
            random.shuffle(gene)
            life = Partical(gene = gene)
            life.cur_score = self.score_cal(gene)
            life.update()
            self.particals.append(life)

        # 进行一次初步的评估
        self.best = self.particals[0]
        self.eval()

    def eval(self):
        """
        进行评估处理,计算适应度信息
        并更新粒子最优和全局最优信息
        """
        for i in range(self.particalCount):
            self.particals[i].cur_score = self.score_cal(self.particals[i].cur_gene)
            self.particals[i].update()
            if self.particals[i].best_score > self.best.best_score:
                self.best = self.particals[i]

    def score_cal(self,gene_item):
        """
        计算一个粒子的适应度信息
        """
        distance = 0.0
        for i in range(-1, self.cityNumber-1):
            index1, index2 = gene_item[i-1], gene_item[i]
            distance += math.sqrt((self.cities_data[index1,0] - self.cities_data[index2,0]) ** 2 + (self.cities_data[index1,1] - self.cities_data[index2,1])** 2)

        return 1/distance

    def cross(self, cur, best):
        """
        进行交叉处理
        """
        one = cur.copy()
        l = [t for t in range(self.cityNumber)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one1 = tmp + cross_part
        l1 = self.score_cal(one1)
        one2 = cross_part + tmp
        l2 = self.score_cal(one2)
        if l1<l2:
            return one1, l1
        else:
            return one2, l2

    def pso_iteration(self):
        """
        进行一轮迭代
        """
        # 1.更新粒子群
        update_paticals = []
        for partical_item in self.particals:
            # 1) 粒子与粒子最优 进行交叉
            # new_gene,new_score = self.cross(partical_item.cur_gene,partical_item.best_gene)
            new_gene, new_score = self.cross(partical_item.cur_gene,partical_item.best_gene)
            #  按一定的概率选择是否接受
            if new_score > partical_item.cur_score or random.random() < 0.1:
                partical_item.cur_gene = new_gene
                partical_item.cur_score = new_score

            # 2)粒子与全局最优粒子 进行交叉
            # new_gene, new_score = self.cross(partical_item.cur_gene, self.best.cur_gene)
            new_gene, new_score = self.cross(partical_item.cur_gene,partical_item.best_gene)
            #  按一定的概率选择是否接受
            if new_score > partical_item.cur_score or random.random() < 0.1:
                partical_item.cur_gene = new_gene
                partical_item.cur_score = new_score
            partical_item.update()
            update_paticals.append(partical_item)
        self.particals = update_paticals

        # 2.评估处理，更新最优信息
        self.eval()

        # 3.返回当先最优信息
        self.generation = self.generation + 1
        return self.best

def cal_tsp(iter_num):
    """
    对tsp问题进行迭代求解
    迭代次数为iter_num
    :return:
    """
    pso = PSO()
    distance_list = []
    best_result = Partical()
    while iter_num > 0:
        pso.pso_iteration()
        distance = 1 / pso.best.best_score
        if iter_num % 50 == 0:
            print(("%d : %f") % (pso.generation, distance))
        distance_list.append(distance)
        iter_num = iter_num - 1


    # 记录损失信息
    loss = np.array(distance_list)
    np.save("resource/pso_loss_1.npy", loss)
    # 输出相关结果
    print("route:", pso.best.best_gene)
    fianl_route = pso.cities_data[pso.best.best_gene]
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
    item_num = 1500
    cal_tsp(item_num)