---
layout:     post
title:      From Global Assessment to Local Selection Efficiently Solving Traveling Salesman Problems of All Sizes
subtitle:   ICLR2025 under review 超大规模TSP 局部改进 两阶段训练
date:       2024/11/01
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
---

From Global Assessment to Local Selection Efficiently Solving Traveling Salesman Problems of All Sizes

从全局评估到局部选择：高效解决各种规模的旅行商问题

ICLR 25 under review

开源： https://anonymous.4open.science/r/ICLR-13204



## 摘要

旅行商问题（TSP）是一个著名的组合优化问题，在现实世界中有着广泛的应用。基于神经网络的TSP求解器的最新进展显示出了令人鼓舞的结果。尽管如此，这些模型通常难以有效地使用相同的预训练模型参数集来解决小型和大型TSP，这限制了它们的实际效用。为了解决这个问题，我们引入了一种名为GELD的新型神经TSP求解器，它建立在我们提出的广泛的全局评估和改进的局部选择框架之上。具体来说，GELD集成了轻量级全局视图编码器（GE）和重量级局部视图解码器（LD），以丰富嵌入表示，同时加快决策过程。此外，GE集成了一种新颖的低复杂度注意机制，使GELD能够实现低推理延迟和更大规模TSP的可扩展性。此外，我们提出了一个两阶段的训练策略，利用不同大小的训练实例来增强GELD的泛化能力。在合成和真实数据集上进行的大量实验表明，考虑到解决方案质量和推理速度，GELD优于七个最先进的模型。此外，GELD可以作为一种后处理方法，以可负担的计算时间换取显着提高的解决方案质量，能够解决多达744,710个节点的TSP，而不依赖于分而治之策略。



## 介绍

现有模型限制：

1. 将预训练的模型泛化到不同大小的TSP通常会导致性能的大幅下降。
2. 神经TSP求解器中常用的标准注意机制的 $O(n^2)$ 时空复杂度限制了其对大规模TSP（如超过1000个节点）的适用性。
3. 通过牺牲计算时间来进一步提升解决方案质量是具有挑战性的，因为神经TSP解算器中使用的神经网络通常作为从节点特征到TSP解的固定映射函数。
4. 虽然基于分治（D&C）策略的模型在解决大规模TSP时表现良好，但它们可能无法为解决其他cop提供有价值的见解，例如作业车间调度问题（JSSP），该问题需要严格的顺序执行并且不易分割。

文章探究的问题：不基于分治的统一预训练模型能否在短时间内有效地解决小型和大型TSP，同时以可承受的时间为代价进一步提高解决方案的质量？

为了回答这个研究问题，引入了一种新的GELD模型，它集成了全局视图编码器（GE）和局部视图解码器（LD）。

回答：

1. GELD是建立在广泛的全局评估和精炼的局部选择框架之上的。具体来说，GELD使用轻量级GE来捕获底层TSP中所有节点的拓扑信息，并与重量级LD配对，在本地选择范围内自回归地选择最有希望的节点。这种双视角方法通过集成全局和局部洞察力来丰富嵌入表示，同时通过将决策空间限制在更小、更相关的子集来加速选择过程，从而提高效率和泛化。
2. 为了降低模型复杂度和进一步加快推理速度，提出了一种新的区域平均线性注意（RALA）机制，该机制在GE内以 $O(n)$ 的时空复杂度运行。RALA将底层TSP中的节点划分为区域，并通过区域代理促进高效的全局信息交换，使GELD能够在短时间内解决TSP问题，并有效地扩展到更大的实例。
3. 为了进一步提高解决方案的质量，将我们提出的多样化模型输入的想法纳入了GELD的架构设计中，使模型不仅可以作为TSP求解器，还可以作为强大的后处理方法，有效地交换可负担的计算时间，以提高解决方案的质量。
4. 为了确保GELD在所有规模的TSP中的鲁棒性，提出了一个两阶段的训练策略，包括不同规模的实例。该方法进一步增强了模型的泛化能力，使其能够使用相同的预训练模型参数集有效地求解TSP。

贡献：

1. 据我们所知，提出的GELD模型是同类中第一个，能够使用相同的预训练模型参数集有效地求解小型和大型TSP。
2. 为GELD提出了一种新颖的低复杂度编码器-解码器骨干架构，实现了低延迟问题解决和可扩展性到更大的TSP实例。
3. 提出了一个两阶段的训练策略，利用不同大小的实例来增强GELD的泛化能力。
4. 展示了GELD作为独立TSP求解器的有效性，以及作为一种强大的后处理方法，通过进行广泛的实验来交换解决方案质量的时间。

## 相关工作

### 神经TSP方法

- 神经构造
  - 自回归：AM、POMO、GNARKD
  - 非自回归
- 神经改进：局部搜索、进化计算（DeepACO）
- 本文既构造又改进

### 神经TSP方法的泛化

- 一般基于分治（有好一些，如GLOP，DISCO），但不适用于其他COP，且忽略子问题之间的相关性，容易导致次优解。

- 扩散模型：DIFUSCO。
- 其他：BQ、LEHD。但他们没法处理超过1000的。

## 预备知识

- TSP设置
- 自回归神经TSP求解器
- 泛化问题
- 用求解时间进一步交换解的质量
  - 数据增强、多次部署、各种搜索
  - 后处理：2-opt、MCTS、beam search（BS）、re-construction（RC）

## GELD: Global-View Encoder and Local-View Decoder

### 架构

![image-20241101115818180]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101115818180.png)

#### Global-View Encoder

为了处理多种不同的分布，对输入坐标归一化。该操作也有利于增强重构的有效性。

$$
\phi(x)=\frac{x-\min_{x_i\in V}(x_i)}{\max_{x_i,x_j\in V}(x_i-x_j)}
$$

然后对输入坐标投影

$$
E=\phi(x)W+b,E\in\mathbb{R}^{n\times h}
$$

使用单个满足下面三个标准的注意力层来提取全局特征

1. 覆盖所有的节点
2. 降低复杂度
3. 以模糊的方式获取全局信息的能力，允许在RC过程中有效地多样化模型输入

现有注意力机制 $E=\text{Softmax}(QK^T)V$ 的时间复杂度 $O(n^2k)$ 和空间复杂度 $O(n^2+nh)$ 很大。

为了满足上述三个标准，提出了区域平均线性注意力（Region-Average Linear Attention, RALA），它以降低的计算复杂度捕捉全局节点特征。

<img src="{{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101115259798.png" alt="image-20241101115259798" style="zoom:50%;" />

具体来说，首先根据归一化的节点坐标将所有节点划分为 $m$ 个区域，记为 $R\_1, \ldots, R\_m$。这里，$m=m\_r \cdot m\_c$ 且 $m \ll n, h$，其中 $m\_r, m\_c \in \mathbb{Z}^{+}$ 表示预定义的行数和列数。推导出的超参数 $m$ 控制区域视图的粒度：更大的 $m$ 值可能捕捉到更多局部区域的见解，但会增加复杂性。然后，使用区域代理促进所有节点之间的全球信息交流，从而满足上述第一标准。通过对该区域内所有节点的查询嵌入 $Q$ 取平均来计算每个区域代理 $P\_i$ 的嵌入，具体如下：

$$
P_i=\left\lbrace \begin{array}{ll}
\frac{1}{n_{R_i}} \sum Q_{x_j}, x_j \in R_i, & \text {if } n_{R_i}>0, \\ 
0_{1 \times h}, & \text {else, }
\end{array} \quad i \in\lbrace 1, \ldots m\rbrace , P \in \mathbb{R}^{m \times h},\right.
$$

其中 $n_{R\_i}$ 表示区域 $R\_i$ 中的节点数量，$Q\_{x\_i} \in \mathbb{R}^{1 \times h}$ 表示节点 $x\_i$ 在查询 $Q$ 中的嵌入。接下来，计算每个区域的节点查询权重分数，具体如下：

$$
Q_w=\operatorname{Softmax}\left(Q P^T\right), Q_w \in \mathbb{R}^{n \times m}
$$

同样，计算区域代理对每个节点的键权重分数，具体如下：

$$
K_w=\operatorname{Softmax}\left(P K^T\right), K_w \in \mathbb{R}^{m \times n}
$$

最后，更新节点特征，以促进全球信息传输，具体如下：

$$
E=Q_w\left(K_w V\right), E \in \mathbb{R}^{n \times h}
$$

与标准注意力机制的平方复杂度不同，RALA在时间和空间复杂度上分别达到 $\mathcal{O}(n m h)$ 和 $\mathcal{O}(n h)$，而无需引入任何额外的可学习参数。这种效率使RALA满足上述第二标准，能够高效地解决大规模实例。此外，在区域划分（RC）过程中，归一化操作的引入导致节点被分配到不同区域以执行RALA。区域代理的多样化更新了全局特征，从而增强了区域划分的有效性，满足了上述第三标准。

#### Local-view Decoder

局部视图解码器中利用多个（精细化的）注意力层。遵循LEHD和BQ采用的解码器设计，根据在MDP步骤 $t$ 中先前选定的节点 $\pi\_{t-1}$ 和目标节点 $\pi\_1$ 的信息，从候选集选择最有前景的节点 $\pi\_t$。与LEHD和BQ考虑所有可用节点作为候选者不同，将候选集限制为节点 $\pi\_{t-1}$ 的 $k$ 最近邻 $K\_{\text{set}}$（即局部选择），其中 $k=\min \lbrace k\_m, n\_t\rbrace $，超参数 $k\_m$ 表示最大局部选择范围，$n\_t$ 表示步骤 $t$ 中剩余可用节点的数量。这种方法减少了决策空间并加快了决策过程。形式上，将节点 $\pi\_{t-1}$ 和 $\pi\_1$ 的特征以及候选集 $K\_{\text{set}}$ 表示为 $E\_{\pi\_{t-1}} \in \mathbb{R}^{1 \times h}$、$E\_{\pi\_1} \in \mathbb{R}^{1 \times h}$ 和 $E\_{K\_{\text{set}}} \in \mathbb{R}^{k \times h}$。将这些特征串联起来，形成MDP步骤 $t$ 的解码器输入，如下所示：

$$
D=\left(E_{\pi_{t-1}}, E_{K_{\text{set}}}, \ldots, E_{\pi_1}\right), D \in \mathbb{R}^{(k+2) \times h} .
$$

为了捕捉局部选择范围内节点之间的细微差别，采用了整合了解码器输入节点之间的距离矩阵 $A$ 的注意力机制。此外，为了减轻由于重复指数运算导致的潜在值溢出，在注意力机制中引入了RMSNorm。解码器中注意力机制的时间和空间复杂度分别为 $\mathcal{O}(k\_m^2 h)$ 和 $\mathcal{O}(k\_m^2 + k\_m h)$。

在通过多个注意力层精炼局部节点特征后，计算MDP步骤 $t$ 中候选集 $K_{\text{set}}$ 的节点被选中的概率分布，如下所示：

$$
p_\theta\left(a_t\right)=\operatorname{Softmax}\left(D_{x_i} W \odot\left\lbrace \begin{array}{ll}
1, & \text{if } x_i \in K_{\text{set}}, \\ 
-\infty, & \text{else},
\end{array}\right), p_\theta\left(a_t\right) \in \mathbb{R}^k,\right.
$$

其中 $D\_{x\_i}$ 表示节点 $x\_i$ 的特征，$\odot$ 表示元素逐位相乘。

### 训练策略

不使用强化学习（RL），其需要大量的计算资源。采用监督学习（SL）、监督增强学习（SIL）。

受到微调大型模型的最新进展的启发，提出了一种两阶段训练方法。

- 第一阶段包括在小规模实例上进行SL训练，然后在更大的实例上进行SIL训练。在第一阶段，利用提供的公开可用的训练数据集，以确保所有相关实验的公平比较。
- 然而，实验结果显示，在小规模TSP上训练的模型在大规模TSP上表现出有限的泛化能力。假设这种限制的出现是因为基于神经网络的模型通常以固定的方式将输入映射到输出。当测试数据中的节点分布与训练数据中的节点分布存在显著差异时，模型难以有效泛化。在这项工作中，在第二阶段扩大了训练数据的大小，以缓解模型在解决更大实例时效率降低的问题。

#### 小规模监督学习

对小于 $k_m$ 的问题定义为小规模。采用交叉熵函数使每一步选择最优动作的概率最大化

$$
\mathcal{L}(\theta | s)=-\sum_{i=1}^n y_i \log (p_\theta(i))
$$

#### 大规模监督增强学习

坚持课程学习策略，逐步将训练实例从小规模的 $k_m$ 扩展到预定义的最大训练大小 $n_{\max}$。

在每个训练 epoch，随机生成一批 $n$ 个训练实例，并应用 beam search 和 PRC(并行RC) 来获得改进的解（在贪婪策略产生的解之上）作为训练的伪标签。当满足以下三个条件之一时，epoch结束：

1. 达到每批训练迭代的最大次数 $t_{max}$；
2. 贪婪解与改进解之间的差距小于预定义的阈值 $\lambda$；
3. 在时间迭代之后，解决方案的质量没有进一步的提高。

为了防止对大规模问题的过拟合，并确保对较小实例的充分关注，在每个 epoch 将10个标记的小规模TSP $k_m$ 实例合并到训练集中。

## 实验结果

四种分布：均匀、聚类、爆炸、内爆

五种规模：100、500、1000、5000、10000

真实数据集：TSPLIB95、National TSPs

合成数据集表现：

![image-20241101152648334]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101152648334.png)

真实数据集表现：

![image-20241101152730647]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101152730647.png)

作为后处理方法重构初始解

![image-20241101152740518]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101152740518.png)

更大的后处理

![image-20241101153935667]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101153935667.png)

消融实验

1. RALA的有效性；
2. 第二阶段训练的影响；
3. GE中通用全局视野的好处；
4. 局部视图在LD中的重要性；
5. 模型输入多样化的有效性。

![image-20241101153903931]({{site.url}}/img/2024-11-01-From_Global_Assessment_to_Local_Selection_Efficiently_Solving_Traveling_Salesman_Problems_of_All_Sizes/image-20241101153903931.png)

## 总结

该方法可以扩展到更复杂的COP如CVRP和JSSP。



























