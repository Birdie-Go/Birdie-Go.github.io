---
layout:     post
title:      A Subpath Ejection Method for the Vehicle Routing Problem
subtitle:   弹射链
date:       2024/12/19
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
---

A Subpath Ejection Method for the Vehicle Routing Problem

1998 Institute for Operations Research and the Management Sciences

里斯本大学



## 省流

结合 TSP Ejection Chains，以下是省流版本。

![image-20241219184110580]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184110580.png)

![image-20241219184120789]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184120789.png)

![image-20241219184135465]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184135465.png)

![image-20241219184144714]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184144714.png)

![image-20241219184153554]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184153554.png)

![image-20241219184204108]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241219184204108.png)



## 摘要

在 CVRP 下的弹射链方法。采用 Flower reference structure。方法基于在弹射链构造过程中创建交替路径和循环的基本性质。文章提出了一种基于禁忌搜索框架的新算法，结果表明该算法可能是VRP最佳启发式算法的一个很好的替代方案。

## 介绍

图 $G=(V,A),V=\lbrace v\_0,\cdots,v\_n\rbrace$，$A$ 是弧的集合，$v\_0$ 是仓库。距离矩阵 $C=(c\_{ij})$。有若干辆车，车数是一个决策变量，每辆车的容量是 $Q$，节点的运货量是 $q\_i$。

该问题的拓展是，卸货需要 $\delta_i$ 的时间，车辆的持续时间不得超过 $D$。

本研究基于禁忌搜索，基本贡献是考虑基于子路径弹出链方法的新邻域结构，用于生成从一个解到一个新解的动作。

## 子路径弹射方法

### The Flower Reference Structure

茎：子路径 $(v\_r,\cdots,v\_c)$，$v\_r$ 是 root，$v\_c$ 是 flower core，$v\_r,v\_c$ 之间的点都只属于一条路径，称之为 blossom。

star：blossom 中的边集 $(v\_c,v\_s)$

简单来说

- 茎表示从 flower core 连出去的一条链，链的另一个端点叫做 root
- blossom 表示从 flower core 的一个环，
- star 就是其中一个端点是 flower core 的边集

一个弹出链过程，描述为，在每一步，一个路由的子路径以一个茎的形式弹出。

图解中，虚线表示可以插入的边，平行线表示可以删除的边，加粗的表示实际选择插入或者删除的边。

![image-20241208182116839]({{site.url}}/img/2024-12-19-VRP_Ejection_Chains/image-20241208182116839.png)

首先要让 VRP 解变成带有一个 stem，其他都是 blossom 的形状。

#### 初始解规则

- rule S1：如图2，为了处理包含单个城市的路线，以及减少给定解的路线总数，可以通过简单地删除属于 star 的边 $(v\_c,v\_j)$ 来获得花结构，从而将 cycle 转化为 stem。
- rule S2：如图2，插入一条边 $(v\_c,v\_i)$，其中 $v\_i$ 不是 $v\_c$ 的相邻顶点，并删除其中一条边 $(v\_i,v\_j)$，从而将一个 cycle 划分为一个 blossom 和一个 stem（多了一个 star）。

两种方式都能让 $v_j$ 变成 root。

#### 生成链

- rule E1：添加一条边 $(v\_r,v\_p)$，其中 $v\_p$ 属于 blossom。选择一条边 $(v\_p,v\_q)$ 删掉。如图3，这样 $v\_q$ 会变成新的 root。如果 $v\_q=v\_c$，那么茎是退化的（就没有茎了），包含了一个单个顶点 $v\_r=v\_c$，包含了一组 blossom。当茎是简并的时候，规则和 S2 相同。
- rule E2：添加一条边 $(v\_r,v\_p)$，其中 $v\_p$ 属于 stem。确定边 $(v\_p,v\_q)$，使得 $v\_q$ 是子路径  $(v\_r,\cdots,v\_p)$ 上的一个顶点。$v\_q$ 变成新的 root。

在生成过程中，root 会一直变，但 core 不会变，core 是仓库。

#### 生成可行解

- Type 1 trial move：将当前的 root $v\_r$ 连接到 star 的其中一个顶点上 $v\_s$，并删除 $(v\_c,v\_s)$。
- Type 2 trial move：添加 $(v\_r,v\_c)$。这样会多一条车辆路径。

事实上，车辆路径数量是会变化的

- S1 + type 1 trial move 会减少路径数量
- S2 + type 2 trial move 会增加路径数量 

省流

- 生成花茎结构
  - 1 删除一个 star
  - 2 添加一个 star，并删除 star 的另一个端点连的其中一条边
- 弹射链
  - 1 root 朝 blossom 的其中一个点连边，并删除边的另一个端点连的其中一条边
  - 2 root 朝 stem 的其中一个点连边，并删除边的另一个端点连的一条边（这条边在这个端点和 root 的路径上）
- 生成可行解
  - 1 root 连到一个 star 的一个端点上，并删除另一条边
  - 2 root 连 core

#### 交替路径考量

在链的每一级使用规则1和规则2有利于生成交替路径，但不能保证交替序列的添加和删除边。举个例子，当 $v\_q$ 成为新的 root $v\_r$ 可以保证从 $v\_r$ 添加的边不在当前解中，但是可以会发生这样一种情况，选择下一个顶点 $v\_q$（$v\_r$ 将与之相连）会导致删除一条边（与新顶点 $v\_q$ 相邻），这条边与另一条不在当前解中的边相邻。

交替路径的考量是：对于局部搜索算法在给定迭代中的解，并非所有边都是“错误的”，即只有一些边不属于最优解。有理由怀疑错误的边在解中是分散的，而不是在解子图的给定区域中定位或聚集。如果只在小范围内改动，一方面这样会非常依赖初始解，另一方面这样无法避免陷入局部最优。

交替路径方法可以避免相邻边在算法的同一步骤同时被修改，但不应该完全禁止。

## 算法

两级 tabu search。

底层 tabu search 通过一个禁忌表，控制被删的边不会被加回来。每个点有一个邻域，定义为最接近它的 $h$ 个节点的集合。对于每个规则的选择，通常在贪心的基础上，增加一个系数来控制不总是选择贪心的策略。算法复杂度 $O(n^2)$。

高层 tabu search 类似 $\varepsilon$ greedy，通过每条边删除和添加的频率，以及一个系数来进行乘法。





