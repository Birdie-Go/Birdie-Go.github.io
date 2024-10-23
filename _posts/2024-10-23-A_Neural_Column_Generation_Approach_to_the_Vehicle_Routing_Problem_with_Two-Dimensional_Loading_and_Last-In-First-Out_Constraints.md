---
layout:     post
title:      A Neural Column Generation Approach to the Vehicle Routing Problem with Two-Dimensional Loading and Last-In-First-Out Constraints
subtitle:   IJCAI2024 神经列生成
date:       2024/10/23
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - IJCAI
---

A Neural Column Generation Approach to the Vehicle Routing Problem with Two-Dimensional Loading and Last-In-First-Out Constraints

二维加载和后进先出约束下车辆路径问题的神经列生成方法

南京大学、1QB信息技术有限公司

IJCAI 2024

代码：https://github.com/xyfffff/NCG-for-2L-CVRP

## 摘要

具有二维载荷约束（2L-CVRP）和后进先出（LIFO）规则的车辆路径问题提出了重大的实践和算法挑战。由于车辆路径问题（VRP）和二维装箱问题（2D-BPP）这两个np困难问题，已经提出了许多启发式方法来解决其复杂性，但很少有人关注开发精确的算法。为了弥补这一差距，本文提出了一种精确的算法，该算法集成了先进的机器学习技术，特别是注意和递归机制的新组合。这种集成将最先进的精确算法在各种问题实例中加速29.79%。此外，所提出的算法成功地解决了标准测试平台中的开放实例，证明了结合机器学习模型带来的显着改进。

## 介绍

VRP -> CVRP -> 2L-CVRP（运输矩形物品，每个物品都有特定的长度、宽度和重量）

2L-CVRP的约束有两个重要的方面：物品的方向（物品是否可以旋转）、顺序加载（后进先出，LIFO）

2L-CVRP包含两个np难问题：CVRP和2DBPP（二维装箱问题）

目前是精确解SOTA是列生成（CG），其特点是反复解决具有挑战性的定价问题。

最近有一些ML和CG结合的研究，现有基于ML的CG通常通过传统方法解决定价问题并生成多个列，ML通常作为列选择的后处理工具。有ML用于2DBPP的定价的方法，但用到2L-CVRP上有困难。与本文相关的一项工作是一个没有后进先出约束的2L-CVRP的变体，利用前馈网络（FF）来加速定价算法生成的列的可行性检查。本研究通过纳入后进先出规则扩展了这种方法，进一步增加了复杂性。

本文提出了一种新的机器学习模型，该模型利用注意机制对同一客户内的同质物品和重复机制对不同客户间的异质物品。该模型用于预测资源约束的基本最短路径问题（ESPPRC）生成的列的可行性，确保符合2D-BPP约束和后进先出规则。本文的方法提供了一种更有效的替代传统的、耗时的可行性检查器，实现了29.79%的中位数加速，并首次成功解决了一个具有挑战性的开放实例。

![image-20241023112653209]({{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023112653209.png)

传统方法的通道图和本文提出方法的通道图如上所示。

贡献：

- 提出了一种神经列生成（NCG）算法，该算法将最先进的2L-CVRP列生成与新开发的ML模型相结合。
- 新机器学习模型集成了注意力和递归机制，以及基于对称的数据增强技术。它通过后进先出规则有效地解决了2D-BPP问题，实现了95%左右的总体精度。
- NCG算法在标准基准测试实例上进行测试时，与最先进的列生成算法相比，运行时间显著减少，中位数减少29.79%。
- NCG算法与最先进的分支-价格-削减（branch-and-price-and-cut， BPC）算法相结合，首次成功解决了一个开放实例。

## 相关工作

- 2L-CVRP的非学习方法（都在20年前）
  - 引入不可行的路径约束，引入使用2L-CVRP的B&C方法，并设置了一个baseline
  - 禁忌启发式搜索，改进了求解时间，并产生了几个最优解
  - 引导局部搜索，改进了求解时间，并产生了几个最优解
  - **启发式目前最有效**：设计模拟退火和局部搜索相结合
  - 列生成解决具有卸货顺序的2L-CVRP
  - 更精细的分治定价，将变量邻域搜索集成到定价中
  - **精确算法重大突破**：B&C结合精确打包算法，引入启发式方法分离分数阶节点上的不可行集不等式，解决几个公开实例并改进对偶边界
  - **最先进的列生成**：采用L-Trie数据结构解决加载约束问题
- 神经列生成（20年后）
  - 采用GNN在CG中进行列选择作为监督二分类任务，该方法将列和约束表示为二分图，CNN预测在CG迭代期间是否包括或者排除每个列
  - 将CG建模为MDP，并用强化学习进行列选择，用GNN作为q函数，证明比贪心有效
  - 基于监督ML，以ESPPRC的形式对定价网络进行修剪，在约简图和完全图之间交替进行，加速CG
  - 通过支持向量机直接生成列来增强CG，解决图染色问题
  - **最相关工作**：解决没有LIFO的VRP，用FF对ESPPRC生成的列进行启发式验证，减少对精确求解器的依赖

## 背景

### 问题建模

2L-CVRP定义在一个完整的无向图 $G=(V, E)$ 上，其中 $V=\lbrace 0,1,2, \ldots, n, n+1\rbrace$ 表示顶点集合，包括客户集合 $V\_c=\lbrace 1,2, \ldots, n\rbrace$ 和仓库0。顶点 $n+1$ 表示仓库的一个副本。任何一对顶点之间的连接由边集 $E$ 描述。对于每条边 $e \in E$，$c\_e$ 表示与边 $e$ 相关的旅行成本。还使用了边 $\left(v\_i, v\_j\right)$ 的另一种表示法。集合 $K$ 表示可用的同质车辆车队。每辆车都有一个以 $H$ 和 $W$ 表示的装载区域，分别为长度和宽度。所有车辆的装载区域相同。显然，每辆车的装载表面总面积为 $A=H \times W$。每辆车还有一个称为 $Q$ 的重量容量。

对于客户，$\forall i \in V\_c$ 由一组矩形 $M\_i$ 特征化。任何物品 $m \in M\_i$ 由宽度 $w\_{i, m}$、长度 $h\_{i, m}$ 和重量 $q\_{i, m}$ 标识。设 $\nu\_i$ 和 $q\_i$ 分别表示客户 $i$ 所有物品的总面积和总重量。换句话说，$\nu\_i=\sum\_{m \in M\_i} w\_{i, m} h\_{i, m}$ 和 $q\_i=\sum\_{m \in M\_i} q\_{i, m}$。在 $G$ 中物品的总数量等于 $\mid M\mid $，其中 $M$ 是所有客户物品集合的并集。2L-CVRP 要求为车队规划路线，以满足客户的需求，同时遵循以下约束：
1. 每位客户必须被访问一次；
2. 车辆运输的物品总重量不得超过 $Q$；
3. 运输的物品必须在装载区内无碰撞地装载；
4. 装载过程中不允许旋转物品；
5. 卸载当前客户的物品时，不允许移动后续客户的物品（LIFO规则）。

2L-CVRP 可以被表述为一个集合划分（SP）问题：

$$
\begin{aligned}
\min & \sum_{r \in \Omega} c_r \lambda_r \\ 
\text { s.t. } & \sum_{r \in \Omega} \lambda_r=\mid K\mid  \\ 
& \sum_{r \in \Omega} a_{i, r} \lambda_r=1, \forall i \in V_c, \\ 
& \lambda_r \in\lbrace 0,1\rbrace, \forall r \in \Omega 
\end{aligned}
$$

其中 $\Omega$ 表示可行路线的集合；$c\_r$ 表示路线 $r$ 的总旅行成本；$\lambda\_r$ 是一个二进制决策变量，指示路线 $r$ 是否被选为解决方案的一部分；$a\_{i, r}$ 是一个二进制指示器，当顶点 $i$ 在路线 $r$ 中被访问时 $a\_{i, r}=1$，否则为 $a\_{i, r}=0$。

方程 (1) 定义了集合划分问题的目标函数。约束集 (2) 规定应选择恰好 $\mid K\mid $ 条路线，因为假设车队中没有闲置车辆。这已成为开发 2L-CVRP 精确算法的惯例。约束集 (3) 规定每位客户必须被访问一次。约束集 (4) 定义了决策变量的域。

直接枚举 $\Omega$ 的所以子集是不可行的，通常选择其一个较小的子集来创建问题的简化版本，称为受限主问题（RMP）。求解 RMP 得到一个解 $\lambda^\ast$，它使简化公式的目标函数值最小，但他可能不算原问题的最优解。为了改进 $\lambda^\ast$，解决一个成为定价问题的子问题，以确定 $\Omega$ 中可以增强解决方案的任何路线。这个迭代过程成为列生成，一直持续到没有发现进一步的改进。

基础定价问题的表述如下：

$$
\begin{aligned}
\min & \sum_{e \in E} \bar{d}_e x_e-\pi_f, \\ 
\text { s.t. } & \sum_{e \in \delta(i)} x_e=2, \forall i \in V, \\ 
& \sum_{e \in \delta(S)} x_e \geq 2, \forall S \subset V_c, 1<\mid S\mid <n-1, \forall i \in S, \\ 
& \sum_{(i, j) \in E} x_{i j}\left(q_i+q_j\right) \leq 2 Q \\ 
& \sum_{(i, j) \in E} x_{i j}\left(\nu_i+\nu_j\right) \leq 2 A, \\ 
& \sum_{e \in E(S, \sigma)} x_e \leq\mid S\mid -1, \forall(S, \sigma) \text { such that } \sigma \notin \Sigma(S), \\ 
& x_e \in\lbrace 0,1\rbrace, \forall e \in E 
\end{aligned}
$$

其中，$\bar{d}\_e$ 是定义为 $\bar{d}\_{i, j}=c\_{i, j}-\frac{1}{2} \pi\_i - \frac{1}{2} \pi\_j, \forall(i, j) \in E$ 的缩减成本。二元决策变量 $x\_e$ 表示边 $e$ 在 $E$ 中是否被使用。集合 $\delta(i)$ 表示与顶点 $i$ 相关的边，$\delta(S)$ 表示连接 $S$ 内部顶点和外部顶点的边，其中 $S$ 是 $V$ 的一个子集。$\Sigma(S)$ 是 $S$ 中所有可行排列的集合。由集合 $S$ 按顺序 $\sigma$ 构造的路径表示为 $(S, \sigma)$，其边集为 $E(S, \sigma)$。对偶变量 $\pi\_i$ 和 $\pi\_f$ 分别与约束 3 和 2 相关联。

目标函数（约束集 5）旨在最小化缩减成本。约束集 (6) 确保顶点的度数约束合理，约束集 (7) 解决了子巡回消除问题。约束集 (8) 和 (9) 确保车辆的容量和装载表面积限制未被超出。最后，约束集 (10) 对装载约束施加限制。

### SOTA pipeline

图1a显示了提出的基于SOTA CG的带有LIFO规则的2L-CVRP算法的流程。该过程从一个受限的主问题开始，该问题本质上是枚举集合 $\omega$ 的一个子集。下一步涉及解决该问题的线性松弛以获得对偶解，从而促进定价的建立。通过trie，完成边界和ng-route松弛增强的标记算法有效地对改进列进行定价，而无需检查加载可行性。最后一步是使用精确检查器过滤掉不可行的列，并将可行的列添加到受限主问题中，迭代直到没有找到可行的列。

## 方法

### 动机和概述

目前的SOTA使用了一个精确的可行性检查器来评估由标记算法生成的每个候选列，这就需要用LIFO规则来解决2D-BPP问题，这是一个已知的np难题。然而，本文提出的方法引入了一个ML模型，通过减少对解决2D-BPP的依赖来改进这一过程。如图1b所示，ML模型位于可行性检查器之前，将候选列分为“可行”和“不可行”。可行的列直接进入受限主问题，而不可行的列进一步检查，以纠正可能的假阴性，从而防止最优列被丢弃。为了管理假阳性，在主问题求解过程中检查最优基中变量的加载可行性，并在可行解空间中加入假阳性切割以排除不可行变量。

这种方法利用了不属于最优基的列可以绕过精确检查的洞察力。然而，它的成功取决于机器学习模型的准确性，因为频繁的错误预测会增加计算迭代。

### ML 模型

**省流：每个客户的物品采用self-attention，客户间采用GRU处理序列，每个客户的物品打乱顺序进行数据增强。**

请注意，标签算法生成的候选列可能违反装载约束。为了解决这个问题，将其框架构建为一个二元分类问题，开发一个机器学习模型来预测每个候选列的可行性。具体来说，模型预测每列满足装载约束的概率。

模型的架构包括一个并行嵌入机制和一个递归处理策略，旨在有效捕捉每列中物品的同质和异质特征。对于给定的输入候选列，表示为物品集的序列 $\left[\left\lbrace x\_{i, m}\right\rbrace\_{1 \leq m \leq\left\mid M\_i\right\mid }\right]\_{1 \leq i \leq n}$，每个物品 $x\_{i, m}$ 通过归一化维度 $\left[\frac{w\_{i, m}}{W}, \frac{h\_{i, m}}{H}\right]$ 表示。对每个客户 $i$ 使用注意力机制来整合同质物品的特征（属于同一客户的物品）。为了处理异质特征（不同客户之间的物品），利用 GRU 来处理序列，结合订单信息，这对于遵循 LIFO 约束至关重要。此外，采用了一种基于对称性的数据增强技术，以将排列不变性融入模型中。如需查看模型架构的可视化表示，请参见图 2。

![image-20241023192415557]({{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023192415557.png)

#### 物品级注意力机制 

在 $2 \mathrm{~L}-\mathrm{CVRP}$ 背景下，属于同一客户的物品是同质的，且不受 LIFO 约束，使得多头注意力（MHA）机制成为合适的选择。模型使用 MHA 捕捉同一客户内部物品的共享特征。第 $i$ 个客户的第 $m$ 个物品的初始嵌入被表述为 $h\_{i, m}^0=W\_0 x\_{i, m}+b\_0$，其中 $W\_0$ 是初始投影矩阵，$b\_0$ 是偏置向量。该架构包括跳跃连接、前馈网络和每个子层的层归一化（LN）。第 $i$ 个客户的第 $m$ 个物品在第 $l$ 层的嵌入被迭代更新，如下方的方程所示：

$$
\begin{gathered}
\hat{h}_{i, m}^l=\mathrm{LN}^l\left(h_{i, m}^{l-1}+\operatorname{MHA}_{i, m}^l\left(h_{i, 1}^{l-1}, \ldots, h_{i,\left\mid M_i\right\mid }^{l-1}\right)\right) \\ 
h_{i, m}^l=\mathrm{LN}^l\left(\hat{h}_{i, m}^l+\mathrm{FF}^l\left(\hat{h}_{i, m}^l\right)\right) 
\end{gathered}
$$

注意力层核心的 MHA 机制定义如下：

$$
\begin{gathered}
Q_{i, m}^j, K_{i, m}^j, V_{i, m}^j=W_Q^j h_{i, m}, W_K^j h_{i, m}, W_V^j h_{i, m} \\ 
A_{i, m}^j=\operatorname{softmax}\left(Q_{i, m}^j K^{j^T} / \sqrt{d_k}\right) V^j \\ 
\operatorname{MHA}_{i, m}=\operatorname{Concat}\left(A_{i, m}^1, A_{i, m}^2, \ldots, A_{i, m}^H\right) W_O 
\end{gathered}
$$

其中 $j=1,2, \ldots, H$ 且 $d\_k=d\_h / H$。这里，$H$ 是注意力头的数量，$d\_h$ 是物品嵌入的维度，$Q\_{i, m}^j, K\_{i, m}^j, V\_{i, m}^j$ 分别代表查询、键和值向量。$W\_O$ 是用于投影最终 MHA 输出的投影矩阵。在 $L$ 层之后每个物品的最终嵌入表示为 $h\_{i, m}=h\_{i, m}^L$。

#### 客户级递归机制 

在遵循 LIFO 规则的 2L-CVRP 背景下，客户之间存在顺序关系，表明不同客户的物品本质上是异质的。这种顺序关系决定了对于客户 $i$，在处理客户 $i+1$ 之前，客户 $i$ 的物品（记作 $\left.x\_{i, 1}, x\_{i, 2}, \ldots, x\_{i,\left\mid M\_i\right\mid }\right)$ 必须在客户 $i+1$ 的物品 $\left(x\_{i+1,1}, x\_{i+1,2}, \ldots, x\_{i+1,\left\mid M\_{i+1}\right\mid }\right)$ 之后装载到车辆中。

为了建模不同客户物品之间的递归关系，使用 GRU，如下所示：

$$
\widetilde{h}_t=\operatorname{GRU}\left(h_t, \widetilde{h}_{t-1}\right),
$$

其中 $\widetilde{h}\_t$ 表示时间步 $t$ 的隐藏状态，且 $\widetilde{h}\_0=\mathbf{0}$。在方法中，客户是按顺序处理的，每个时间步 $t$ 输入一个物品 $h\_t$ 到 GRU，总时间步数 $T$ 等于总物品数，即 $T=\sum\_{i=1}^n\left\mid M\_i\right\mid $。在处理完最后一个客户的最后一个物品后，GRU 的最终状态 $\widetilde{h}\_T$ 通过前馈网络和 sigmoid 函数转化为概率：

$$
\text { probability }=\operatorname{sigmoid}\left(\mathrm{FF}\left(\widetilde{h}_T\right)\right) .
$$

#### 具有排列不变性的数据增强

属于同一客户的物品是同质的，不受任何特定顺序的约束。这一特性使得排列不变性作为数据增强策略得以应用，反映了组合问题的对称性。通过对每个客户的物品进行排列，可以生成新的等效实例，扩展训练数据集并减轻早期过拟合。

具体而言，在对每个客户 $i$ 的所有物品应用物品级多头注意力机制后，我们执行排列 $\pi\_i$，打乱序列 $\left(1,2, \ldots,\left\mid M\_i\right\mid \right)$ 以产生不同的物品顺序。这个过程可以数学上表示为：

$$
h_{i, 1}, h_{i, 2}, \ldots, h_{i,\left\mid M_i\right\mid }=h_{i, \pi_i(1)}, h_{i, \pi_i(2)}, \ldots, h_{i, \pi_i\left(\left\mid M_i\right\mid \right)}
$$

这个排列在处理 GRU 之前独立应用于每个客户的物品集合。

## 实验

### 实验设置

- 数据集：PP实例来自 2L-VRPTW 和 2L-CVRP，VPR实例是随机生成的。
- 验证集：求解公式的线性松弛来比较NCG和SOTA。
- 硬件：单卡A100

### 实验结果

加速比：测试对象有50个客户，平均性能提升 29.79\%。3个数据集的提升分别为，9.38\%、44.97\%、99.22\%。

![image-20241023194606608]({{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023194606608.png)

在公开数据集上表现很好，成功解决了公开实例：

![image-20241023194117031]({{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023194117031.png)

### 消融实验

- 完整的方法
- 没有数据增强
- 用MLP代替注意力机制
- 正弦位置编码的Transformer encoder 代替 GRU

![image-20241023194326158]({{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023194326158.png)

数据增强能够避免过拟合

<img src="{{site.url}}/img/2024-10-23-A_Neural_Column_Generation_Approach_to_the_Vehicle_Routing_Problem_with_Two-Dimensional_Loading_and_Last-In-First-Out_Constraints/image-20241023195015075.png" alt="image-20241023195015075" style="zoom:50%;" />

## 总结

主要贡献是：显著加速了SOTA算法的中位数29.79\%。





