---
layout:     post
title:      Memory-Enhanced Neural Solvers for Efficient Adaptation in Combinatorial Optimization
subtitle:   Arxiv2024.10 经验池
date:       2025/04/25
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
---

Memory-Enhanced Neural Solvers for Efficient Adaptation in Combinatorial Optimization

Arxiv 2024.10.7

InstaDeep

开源：[instadeepai/memento: Official Implementation of Memento](https://github.com/instadeepai/memento)

## 摘要

组合优化对许多现实世界的应用至关重要，但由于其（NP-）困难的性质，仍然存在挑战。在现有的方法中，优化通常在质量和可扩展性之间提供最佳权衡，使其适合工业使用。而强化学习（RL）为设计优化提供了灵活的框架，现有的学习方法仍然缺乏适应特定实例的能力，并且不能充分利用可用的计算预算。目前最好的方法要么依赖于一系列预先训练的策略，或数据效率低下的微调;因此，未能充分利用新的可用信息的预算的限制。作为回应，我们提出MEMEMENTO，一种方法，利用记忆来提高神经求解器在推理时间的适应性。MEMEMENTO能够根据以前的决策结果动态更新动作分布。我们验证了其在基准问题上的有效性，特别是旅行推销员和容量限制车辆路径，证明了它的优越性，树搜索和政策梯度微调，并显示它可以与基于多样性的解算器零杆相结合。

我们成功地在大型实例上训练了所有RL自回归求解器，并表明MEMENTO可以扩展并且具有数据效率。总体而言，MEMENTO能够在12个评估任务中的11个上推动最先进的技术。



## 方法

### 出发点

将所有过去的尝试存储在一个内存中，可以用于后续的轨迹。这确保了没有信息丢失，并且有希望的轨迹可以使用多次。

希望基于内存的更新机制是

1. 可学习的（应当学习如何使用过去的轨迹来制作更好的轨迹，而不是进行硬编码）
2. 轻量化以不过度地损害推理时间
3. 与底层模型架构无关
4. 能够利用预先训练的无记忆策略。

### 总览

![image-20250425153439298]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425153439298.png)

引入了MEMENTO，这是一种使用在线收集的数据动态调整神经求解器的动作分布的方法。

可应用到：AM、POMO、Poppy、COMPASS中。

### 存储数据到内存

类似经验池，可进行经验回放。存储：

1. 访问的节点
2. 动作，对应于策略决定下一次访问的节点的概率
3. 模型给出该动作的对数概率
4. 整个轨迹的负成本
5. 构建方案的时间成本
6. 内存中建议给出的该动作的动作对数概率
7. 与整个轨迹相关联的动作概率
8. 轨迹剩余部分的动作概率

2 3 4 是重现REINFORCE更新所需的部分，其余是有助于信用分配和分配偏移估计的附加上下文。

### 内存中检索数据

由于这种检索过程将在每一步发生，不希望它成本太高。

通过检索当前所在的同一节点中收集的数据，可以实现良好的速度/相关性平衡。

比k-最近邻检索更快，同时提取相似相关性的数据。

### 处理数据以更新动作

![image-20250425160929857]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425160929857.png)

以下是“Processing data to update actions”部分的中文翻译：

一旦检索到数据，便将其用于更新基础策略的动作概率。从数据中，我们将动作与其相关特征（对数概率、回报等）分开。将剩余预算附加到特征上。每个特征被归一化后，得到的特征向量由一个多层感知机（MLP） $ H\_{\theta\_M} $ 处理，输出一个标量权重。每个动作被独热编码，基于获得的标量权重计算动作向量的加权平均值。这个聚合操作输出一个新的动作概率向量。这个向量与基础模型输出的动作概率向量相加。接下来介绍数学形式化内容，图2对其进行了说明。

更正式地说，当访问一个节点时，给定状态 $ s $ ，我们从记忆中检索过去访问该节点时的数据。这些数据 $ M\_s $ 是一系列元组 $ (a\_i, f\_{a\_i}) $ ，其中 $ a\_i $ 是过去尝试的动作， $ f\_{a\_i} $ 是与相应轨迹相关的各种特征，如上所述。更新计算公式为 $ l\_M = \sum\_i a\_i H\_{\theta\_M}(f\_{a\_i}) $ ，其中 $ a\_i $ 是动作 $ a\_i $ 的独热编码。设 $ l $ 为基础策略的logits，则用于采样下一个动作的最终logits为 $ l + l\_M $ 。能证明，最坏情况下，这使得MEMENTO能够重新发现REINFORCE更新。

（通过经验池中的历史数据来校正当前输出的动作概率）

### 训练

对于每个问题实例，都希望优化最佳回报。在每次尝试中，我们对最后的回报与迄今为止获得的最佳回报之间的差值应用修正线性单元（ReLU）函数。我们使用修正后的差值来计算每次尝试的REINFORCE损失，以避免奖励过于稀疏，并通过网络参数（包括编码器、解码器和记忆网络的参数）进行反向传播。

具体而言，给定一个问题实例，我们展开  $ B $  条轨迹，并将它们依次存储在记忆中。每条轨迹  $ \tau\_i $  产生一个回报  $ R(\tau\_i) $ 。每条轨迹的优势定义为  $ \tilde{R}(\tau\_i) = \max(R(\tau\_i) - R\_{\text{best}}, 0) $ ，其中  $ R\_{\text{best}} $  是迄今为止找到的最高回报。用于更新策略的总损失是通过REINFORCE算法计算的：

$$
L = -\sum_{i=1}^{B} \log(1 + \epsilon + i) \tilde{R}(\tau_i) \sum_{t} \log \pi_M(a_t|s_t, M_t),
$$

其中  $ \pi\_M $  是通过logits  $ l\_M $ （如文中公式1所示）丰富后的策略， $ \epsilon $  是一个小常数，确保第一项不为零。

为了使计算保持可行，我们仍然在每一步计算损失，估计梯度，并将它们顺序累加，直到达到预算，然后进行一次梯度更新步骤，并考虑一个新的实例批次。

## 实验

在TSP和CVRP中评估。

baseline：

- 使用在线收集的数据的单一政策适应的领先方法EAS
- SOTA神经求解器COMPASS

![image-20250425162501034]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425162501034.png)

EAS和MEMENTO的基础都是POMO。

![image-20250425162648709]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425162648709.png)

观察到MEMENTO学习的主要规则与REINFORCE相似：一个低logp和高回报的行为得到一个积极的更新，而一个高logp和低回报的行为得到劝阻。尽管如此，看到差异还是很有趣的。特别是，MEMEMENTO的规则关于x = 0是不对称的，它只鼓励高于平均回报的行为，但对高回报行为的激励幅度高于对低回报行为的抑制，在保持探索性的同时推动了绩效。EAS也会出现类似的不对称，因为它将REINFORCE与一个增加生成最佳解决方案的可能性的术语相结合。

### MEMENTO与不可见解算器的zero-shot组合

![image-20250425162802840]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425162802840.png)

![image-20250425162742250]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425162742250.png)

### 泛化到更大规模

![image-20250425163130862]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425163130862.png)

![image-20250425163142033]({{site.url}}/img/2025-4-25-Memory-Enhanced_neural_solvers_for_efficient_adaptation_in_combinatorial_optimization/image-20250425163142033.png)





