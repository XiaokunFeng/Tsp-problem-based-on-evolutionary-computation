# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
"""
description: 使用 模拟退火算法 来求解 中国34个城市的TSP问题
              不使用 Grefenstette 编码
"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
class Life(object):
    """
    个体类
    """
    def __init__(self, gene=None):
        self.gene = gene                    # 个体基因信息
        self.score = 0                      # 个体适应度 = 1/dis
        self.f = 0                          # 根据温度求解得到的适应度


class GA(object):
    """
    遗传算法类
    """
    def __init__(self, lifeCount=100,crossRate=0.9, mutationRate=0.05,selectRatio=0.2,cities_path="resource/china_cities.npy"):
        self.lifeNumber = lifeCount                      # 个体数量
        self.crossRate = crossRate                      # 交叉率
        self.mutationRate = mutationRate                # 变异率
        self.selectRatio = selectRatio                  # 子代中保留个体的比例
        self.selectedLives_num = int(self.lifeNumber * self.selectRatio)

        self.lives = []                                 # 各个个体信息的集合
        self.generation = 0                             # 进化代数
        self.best = None                                # 当前代数下，最优的个体
        ## 新引入的变量
        self.T0 = 4000
        self.A = 0.9995

        # 初始化城市地点信息
        self.cities_data = np.load(cities_path)
        self.cityNumber = self.cities_data.shape[0]
        # 初始化种群数据
        self.initialGeneration()

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
            life = Life(gene = gene)
            life.score =self.score_cal(gene)
            life.f = np.exp(life.score / self.T0)
            self.lives.append(life)
        # 找到最优值
        self.best = self.lives[0]
        for i in range(self.lifeNumber):
            if self.lives[i].score > self.best.score:
                self.best = self.lives[i]
    def initialGeneration_by_input_data(self,initial_data_path):
        """
        通过给定数据的方式来初始化种群信息
        初始化种群信息
        """
        # 为种群中的各个个体赋值
        initial_data = np.load(initial_data_path)
        initial_data = initial_data.tolist()
        self.lives = []
        for i in range(self.lifeNumber):
            gene = initial_data[i]
            life = Life(gene=gene)
            life.score = self.score_cal(gene)
            self.lives.append(life)
        # 找到最优值
        self.best = self.lives[0]
        for i in range(self.lifeNumber):
            if self.lives[i].score > self.best.score:
                self.best = self.lives[i]

    def score_cal(self,gene_item):
        """
        计算一个个体的适应度信息
        """
        distance = 0.0
        for i in range(-1, self.cityNumber-1):
            index1, index2 = gene_item[i-1], gene_item[i]
            distance += math.sqrt((self.cities_data[index1,0] - self.cities_data[index2,0]) ** 2 + (self.cities_data[index1,1] - self.cities_data[index2,1])** 2)
        return 1/distance

    def eval(self):
        """
        进行评估处理
        """
        self.T0 = self.T0 * self.A
        for i in range(self.lifeNumber):
            self.lives[i].score = self.score_cal(self.lives[i].gene)
            self.lives[i].f = np.exp(self.lives[i].score/self.T0)
            if self.lives[i].score > self.best.score:
                self.best = self.lives[i]

    def select_process(self):
        """
        从当前代种群中，选择一定比例的子代
        """
        # 1.先统计各个个体的得分情况
        lives_f_list = []
        for life_item in self.lives:
            lives_f_list.append(life_item.f)
        # 2.排序和选择
        lives_f_list = np.array(lives_f_list)
        sort_index = np.argsort(-1*lives_f_list).copy()
        select_index = sort_index[0:self.selectedLives_num]
        selectedLives = []
        for index in select_index:
            selectedLives.append(self.lives[index])
        self.lives = selectedLives

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

    def cross_and_mutation(self):
        """
        在选择得到的个体的基础上，进行交叉变异产生子代
        直到将种群的数量增至 self.lifeNumber
        """
        # 1.先根据 已选择个体的score,求出对应的概率信息
        probs = []
        for life_item in self.lives:
            probs.append(life_item.f)
        probs = np.array(probs)
        probs = probs/np.sum(probs)

        # 2.进行交叉 和 变异 处理
        while len(self.lives) < self.lifeNumber:
            # 选择两个父类
            parent1_index = self.select_roulette(probs)
            parent2_index = self.select_roulette(probs)
            while parent2_index == parent1_index:
                parent2_index = self.select_roulette(probs)
            parent1,parent2 = self.lives[parent1_index],self.lives[parent2_index]
            # 按概率交叉
            if np.random.rand() < self.crossRate:
                child1_gene,child2_gene = self.cross(parent1, parent2)
            else:
                child1_gene = parent1.gene
                child2_gene = parent2.gene

            # 按概率突变
            if np.random.uniform() < self.mutationRate:
                # child1_gene = self.mutation(child1_gene)
                child1_gene = self.mutation_2(child1_gene)
            if np.random.uniform() < self.mutationRate:
                # child2_gene = self.mutation(child2_gene)
                child2_gene = self.mutation_2(child2_gene)

            # 3.将适应度高的放入种群
            child1_score = self.score_cal(child1_gene)
            child2_score = self.score_cal(child2_gene)
            if child1_score > child2_score:
                child = Life(child1_gene)
                child.score = child1_score
            else:
                child = Life(child2_gene)
                child.score = child2_score
            self.lives.append(child)

    def cross(self, parent1, parent2):
        """
        交叉
        """
        # 双点交叉
        index1 = random.randint(0, self.cityNumber - 1)
        index2 = random.randint(index1, self.cityNumber - 1)

        # 子代1
        tempGene = parent2.gene[index1:index2]
        newGene1 = []
        p1len = 0
        for g in parent1.gene:
            if p1len == index1:
                newGene1.extend(tempGene)
                p1len += 1
            if g not in tempGene:
                newGene1.append(g)
                p1len += 1

        # 子代2
        tempGene = parent1.gene[index1:index2]
        newGene2 = []
        p1len = 0
        for g in parent2.gene:
            if p1len == index1:
                newGene2.extend(tempGene)
                p1len += 1
            if g not in tempGene:
                newGene2.append(g)
                p1len += 1

        return newGene1,newGene2
    def mutation(self, gene):
        """
        突变
        """
        index1 = random.randint(0, self.cityNumber - 1)
        index2 = random.randint(0, self.cityNumber - 1)

        newGene = gene.copy()
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        return newGene
    def mutation_2(self, gene):
        """
        第2种 变异的方式
        两个变异点之间的数据也跟着倒置
        """
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga_iteration(self):
        """
        进行一次迭代运算
        """
        # 1.选择操作
        self.select_process()
        # 2.交叉变异
        self.cross_and_mutation()
        # 3.进行一次评估更新
        self.eval()

        self.generation = self.generation + 1
        return self.best

def cal_tsp(iter_num):
    """
    对tsp问题进行迭代求解
    迭代次数为iter_num
    """
    ga = GA(lifeCount=100,crossRate=0.9, mutationRate=0.05,selectRatio=0.2,cities_path="resource/china_cities.npy")
    # 通过给定数据来完成初始化(为了对比实验)
    # initial_data_path = "resource/initial_data.npy"
    # ga.initialGeneration_by_input_data(initial_data_path)

    distance_list = []
    while iter_num > 0:
        ga.ga_iteration()
        distance = 1 / ga.best.score
        if iter_num%50==0:
            print(("%d : %f") % (ga.generation, distance))
        distance_list.append(distance)
        iter_num = iter_num - 1

    loss = np.array(distance_list)
    np.save("resource/loss_4.npy", loss)
    # 输出相关结果
    print("route:", ga.best.gene)
    fianl_route = ga.cities_data[ga.best.gene]
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
    plt.text(iterations[-1]-20, distance_list[-1]+10, '%.1f' % distance_list[-1], fontdict={'fontsize': 14})
    plt.title('Distance')
    plt.show()

if __name__ == "__main__":
    item_num = 500
    cal_tsp(item_num)