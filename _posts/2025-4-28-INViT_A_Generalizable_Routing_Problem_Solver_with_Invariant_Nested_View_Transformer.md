---
layout:     post
title:      INViT A Generalizable Routing Problem Solver with Invariant Nested View Transformer
subtitle:   ICML2024 大规模+多分布
date:       2025/04/28
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - ICML
---

INViT: A Generalizable Routing Problem Solver with Invariant Nested View Transformer

上海交通大学密歇根联合研究所，昆山杜克大学

ICML2024

开源：[Kasumigaoka-Utaha/INViT: Official Implementation of the paper: INViT: A Generalizable Routing Problem Solver with Invariant Nested View Transformer](https://github.com/Kasumigaoka-Utaha/INViT)



## 摘要

最近，深度强化学习在学习快速算法来解决路由问题方面显示出了很好的效果。同时，大多数求解器都受到推广到未知分布或不同尺度分布的困扰。为了解决这个问题，我们提出了一种新的架构，称为不变嵌套视图Transformer（INViT），它的设计是为了在编码器内部执行嵌套设计和不变视图，以提高学习求解器的泛化能力。它采用了一种改进的策略梯度算法，并通过数据增强来增强。我们结果表明，所提出的INViT在不同分布和不同问题规模的TSP和CVRP问题上都取得了显著的泛化性能。



## 引入

贡献

- 确定了两个因素来解释大多数基于DRL的方法中观察到的泛化问题：嵌入别名和来自不相关节点的干扰。通过分析路由问题最优解的一些统计特性，我们激励减少状态和动作空间。

- 设计了一个新的基于Transformer的架构，它采用路由问题实例的不变嵌套视图。它的架构是由我们以前的观察和统计分析证明的。

- 在不同的数据集上证明，所提出的架构在TSP和CVRP的泛化方面优于当前的SOTA方法。



## 动机

为了设计一个可以在交叉大小和交叉分布设置中很好地泛化的自回归求解器，首先确定了当前在小规模和均匀分布实例上训练的神经求解器的两个缺点：嵌入混叠和来自不相关节点的干扰。

- 根据Lipschitz不等式推导出，编码器对于泛化到更大规模的问题，会导致求解器进行不正确的动作选择。
- 对于询问节点，需要计算出对所有节点的注意力分数，当规模比较大的时候，那些分数较低的节点的分数和的累积影响变得不可忽略。

![image-20250428134806602]({{site.url}}/img/2025-4-28-INViT_A_Generalizable_Routing_Problem_Solver_with_Invariant_Nested_View_Transformer/image-20250428134806602.png)

（a）基于注意力的编码器（在TSP/CVRP 100上训练）中更远节点的注意力得分直方图。（B）不同k的k次最近邻中最优解的直方图。（c）原始实例和增强实例之间最优解的重叠百分比。



## 方法

Invariant Nested View Transformer (INViT)。

![image-20250426163657001]({{site.url}}/img/2025-4-28-INViT_A_Generalizable_Routing_Problem_Solver_with_Invariant_Nested_View_Transformer/image-20250426163657001.png)

### 嵌套视图编码器

对于静态图，计算稀疏图是一个可行的任务。然而，考虑到路径问题的动态性，在推理过程中动态计算稀疏图是一个计算成本高昂的任务。因此，我们提出了一个嵌套视图编码器设计，通过将图稀疏化为子图来解决这一问题，每个子图由不同数量的邻居组成。由于 k-最近邻（k-NN）算法可以提供稳定的邻居，并且可以批量操作，我们使用它来进行图稀疏化。通过消除不同 k 值下不在邻居中的节点，可以生成多个子图。经过不变层处理后，每个并行的单视图编码器将接收一个不同的不变子图，并在不同的图上下文中输出嵌入。

### 不变层

由于节点直接距离过小，编码器就会很难分辨。所以设计了一个不变层，将节点归一化。

不变层包括两个步骤：归一化和投影。归一化可以表示为：

$$
\hat{c}_{mi} = \frac{c_{mi} - \min_{j \in N} c_{mj}}{\max_{m \in \{1,2\}} \max_{i,j \in N} |c_{mi} - c_{mj}|}
$$

除了潜在候选集  $  A\_p  $  和最后访问的节点外，子图还包括第一个访问的节点（或仓库），其影响不容忽视。然而，第一个访问的节点（或仓库）可能位于潜在候选集  $  A\_p  $  和最后访问节点的区域之外，可能会削弱归一化过程的有效性。在这种情况下，我们引入了一个投影步骤，可以表示为：

$$
\hat{c}_{m0} = \text{clip} \left( \frac{c_{m0} - \min_{j \in N} c_{mj}}{\max_{m \in \{1,2\}} \max_{i,j \in N} |c_{mi} - c_{mj}|}, 0, 1 \right)
$$

其中  $  \text{clip}(u, v, w) = \max(v, \min(u, w))  $  将区域外的第一个访问节点（或仓库）投影到边界上，确保其坐标的边界不变。

### 单视图编码器

不使用位置编码，跟节点的输入顺序没关系。编码器同AM和POMO。

### 多视图解码器

通过通道级联将多个单视图嵌入聚合，然后输入到多视图解码器中。解码器同AM和POMO。

### 训练

同REINFORCE，对每个实例使用了数据增强。

### 测试

测试阶段也使用了数据增强。

## 实验

数据集

- 生成数据集：MSVDRP（多尺度多分布RP），TSP包含了4种规模（100,1000,5000,10000）和4种分布（均匀、聚类、爆炸、内爆）的16种组合，CVRP包含了3种规模（50,500,5000）和4种分布的12种组合。
- 公开数据集：TSPLIB和CVRPLIB
- 评估标准：LKH、Gurobi或者HGS生成baseline

实验设置

- INViT-2V包含2个编码器（k=15,35），3V包含3个编码器（k=15,35,50）。每个编码器有两层。
- pomo sample最大为100。
- 单个Intel Core i7-12700 CPU和单个RTX 4090 GPU。

生成数据集表现

![image-20250428142654692]({{site.url}}/img/2025-4-28-INViT_A_Generalizable_Routing_Problem_Solver_with_Invariant_Nested_View_Transformer/image-20250428142654692.png)

公开数据集表现

![image-20250428143108533]({{site.url}}/img/2025-4-28-INViT_A_Generalizable_Routing_Problem_Solver_with_Invariant_Nested_View_Transformer/image-20250428143108533.png)

### 消融实验

包括

- Global，其中一个编码器处理全局信息而不稀疏化
- w/o Inv，没有不变层
- head=4，原本8头
- aug=4，原本8
- w/o Aug，不用数据增强
- Model-50，在50的实例上训练，原本是100
- 1V，50,35,15里面挑一个最好的
- 4v，加一个75

![image-20250428143158799]({{site.url}}/img/2025-4-28-INViT_A_Generalizable_Routing_Problem_Solver_with_Invariant_Nested_View_Transformer/image-20250428143158799.png)











