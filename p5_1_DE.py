# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
"""
description: 使用基本的差分进化优化算法来求解 中国34个城市的TSP问题
"""
import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

class Life(object):
    """
    个体类
    """
    def __init__(self, gene=None):
        """
        对个体进行初始化
        """
        self.gene = gene                   # 个体信息
        self.score = 0                     # 个体的适应度


class DE(object):
    """
    差分算法类
    """
    def __init__(self, lifeNumber=100,cities_path="resource/china_cities.npy"):
        self.lifeNumber = lifeNumber                      # 个体数量
        self.Factor = 0.4  # 缩放因子
        self.crossRate = 0.5  # 交叉概率

        self.generation  = 0
        # 初始化城市地点信息
        self.cities_data = np.load(cities_path)
        self.cityNumber = self.cities_data.shape[0]

        self.lives = []                                 # 种群集合
        self.best = None                                # 当前代数下，最优的个体

        # 初始化种群
        self.initialGeneration()

        self.permute_index = 1

    def initialGeneration(self):
        """
        初始化种群信息
        :return:
        """
        # 为种群中的各个个体赋值
        self.lives = []
        for _ in range(self.lifeNumber):
            gene = []
            for gene_item in range(self.cityNumber):
                gene.append(gene_item)
            # 乱序处理
            random.shuffle(gene)
            life = Life(gene=gene)
            life.score = self.score_cal(gene)
            self.lives.append(life)
        # 求best
        self.best = self.lives[0]
        for life_item in self.lives:
            if life_item.score > self.best.score:
                self.best = life_item

    def score_cal(self,gene_item):
        """
        计算适应度信息
        """
        distance = 0.0
        for i in range(-1, self.cityNumber-1):
            index1, index2 = gene_item[i-1], gene_item[i]
            distance += math.sqrt((self.cities_data[index1,0] - self.cities_data[index2,0]) ** 2 + (self.cities_data[index1,1] - self.cities_data[index2,1])** 2)

        return 1.0/distance

    def select_3_lives(self,index):
        """
        选择出 除了 index 外的3个个体
        """
        r0,r1,r2 =-1,-1,-1
        while r0 == index or r0==-1:
            r0 = np.random.randint(self.cityNumber - 1)

        while r1 == index or r1 == r0 or r1 == -1:
            r1 = np.random.randint(self.cityNumber - 1)

        while r2 == index or r2 == r0 or r2 == r1 or r2 == -1:
            r2 = np.random.randint(self.cityNumber - 1)
        return self.lives[r0].gene,self.lives[r1].gene,self.lives[r2].gene

    def select_2_lives(self,index,r0):
        """
        选择出 除了 index 外的3个个体
        """
        r1,r2 =-1,-1

        while r1 == index or r1 == r0 or r1 == -1:
            r1 = np.random.randint(self.cityNumber - 1)


        while r2 == index or r2 == r0 or r2 == r1 or r2 == -1:
            r2 = np.random.randint(self.cityNumber - 1)
        return self.lives[r1].gene,self.lives[r2].gene

    def transform(self,u_gene):
        """
        将计算得到的结果，转换成符合TSP问题解的形式
        """
        u_gene_arr = np.array(u_gene)
        # 避免可能出现两个元素相等的情况
        for u_item in u_gene:
            u_item_index = np.where(u_gene_arr == u_item)[0]
            if len(u_item_index) >1:
                add_scale = 0.001
                for u_item_index0 in u_item_index:
                    u_gene[u_item_index0] = u_gene[u_item_index0] + add_scale
                    add_scale =add_scale*0.1

        u_sorted = sorted(u_gene.copy())
        u_gene_update = []
        for city in u_gene:
            u_gene_update.append(u_sorted.index(city))

        for i in range(self.cityNumber):
            if i not in u_gene_update:
                print("UnFeasible!")
                print(u_gene)
                print(len(u_gene_update))
                print(u_gene_update)
                print(min(u_gene_update))
                print(max(u_gene_update))
                sys.exit()

        return u_gene_update

    def cross(self, cur, best):
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

    def get_u_by_dim_item(self,life_item,life0,life1,life2):
        """
        通过对每一维进行分析，得到测试个体u
        """

        j_rand = np.random.randint(self.cityNumber - 1)
        # 对每一维进行分析，进行交叉处理
        u_gene = []
        for j in range(self.cityNumber):
            if random.random()<self.crossRate or j == j_rand:
                u_item = life0[j] + self.Factor*(life1[j]-life2[j])
                u_gene.append(u_item)
            else:
                u_gene.append(life_item.gene[j])
        # 将计算得到的结果，转换成符合TSP问题解的形式
        u_gene = self.transform(u_gene)
        return u_gene

    def get_u_by_cross(self, life_item, life0, life1, life2):
        """
        通过交叉，得到测试个体u
        """
        u_gene1, score1 = self.cross(life0, life1)
        u_gene2, score2 = self.cross(life0, life2)
        u_gene3, score3 = self.cross(life1, life2)

        if score1 > score2:
            u_gene = u_gene1
            u_gene_score = score1
        else:
            u_gene = u_gene2
            u_gene_score = score2

        if score3 > u_gene_score:
            u_gene = u_gene3
            u_gene_score = score3
        return u_gene

    def de_iteration(self):
        # 1.确定 目标对应的基向量索引
        # 确保每个个体只能成为基向量1次
        permute = []
        self.permute_index = self.permute_index + 1
        if self.permute_index >= self.lifeNumber:
            self.permute_index = 1
        for i in range(self.lifeNumber):
            permute.append((i+self.permute_index)%self.lifeNumber)

        # 2.对种群中的每一个个体进行交叉变异处理
        for life_index in range(self.lifeNumber):
            # 1) 先选出3个其他个体
            life_item = self.lives[life_index]
            life0_index = permute[life_index]
            life0 = self.lives[life0_index].gene
            life1, life2 = self.select_2_lives(life_index,permute[life_index])
            # life0,life1,life2 = self.select_3_lives(life_index)

            # 2) 交叉产生u
            # u_gene = self.get_u_by_dim_item(life_item,life0,life1,life2)
            # 另一种交叉方式
            u_gene = self.get_u_by_cross(life_item, life0, life1, life2)

            # 3. 选择处理
            u_gene_score = self.score_cal(u_gene)
            if u_gene_score > life_item.score:
                life_item.gene = u_gene
                life_item.score = u_gene_score
                self.lives[life_index] = life_item

        # 4.计算当前最优解
        for life_item in self.lives:
            if life_item.score > self.best.score:
                self.best = life_item
        self.generation = self.generation + 1
        return self.best

def cal_tsp(iter_num):
    """
    对tsp问题进行迭代求解
    迭代次数为iter_num
    :return:
    """
    de = DE()
    distance_list = []
    while iter_num > 0:
        best_partical = de.de_iteration()
        distance = 1 / best_partical.score
        if iter_num%50 == 0:
            print(("%d : %f") % (de.generation, distance))
        distance_list.append(distance)
        iter_num = iter_num - 1
    # 记录损失信息
    loss = np.array(distance_list)
    np.save("resource/de_loss_1.npy", loss)

    # 输出相关结果
    print("route:", de.best.gene)
    fianl_route = de.cities_data[de.best.gene]
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