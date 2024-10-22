---
layout:     post
title:      Towards Generalizable Neural Solvers for Vehicle Routing Problems via Ensemble with Transferrable Local Policy
subtitle:   IJCAI2024 局部策略 泛化能力
date:       2024/10/22
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - IJCAI
---

Towards Generalizable Neural Solvers for Vehicle Routing Problems via Ensemble with Transferrable Local Policy

南京大学，华为诺亚方舟实验室

IJCAI 2024

代码：https://github.com/lamda-bbo/ELG

## 摘要

机器学习已经被用来帮助解决NP-hard组合优化问题。一种流行的方法是利用深度神经网络来学习构造解，这种方法由于效率高、对专家知识要求低而受到越来越多的关注。然而，许多车辆路径问题（Vehicle Routing Problems, VRPs）的神经网络构建方法都集中在具有特定节点分布和有限规模的综合问题实例上，导致该方法在涉及复杂未知节点分布和大规模的现实问题上表现不佳。为了使神经VRP求解器更加实用，我们设计了一个从局部可转移拓扑特征中学习的辅助策略，称为局部策略，并将其与典型的构造策略（从VRP实例的全局信息中学习）相结合，形成一个集成策略。通过联合训练，聚合策略协同互补，提高泛化能力。在旅行商问题和有容量VRP的两个著名基准TSPLIB和CVRPLIB上的实验结果表明，集成策略显著提高了交叉分布和跨尺度泛化性能，甚至在具有数千个节点的现实问题上也有很好的表现。

## 介绍

与以往依赖于分而治之策略或尝试改进学习算法的方法不同，文章提出了一种集成方法，该方法集成了从vrp的全局信息中学习的策略和具有可转移性的局部策略，其中它们的优势相结合以提高泛化。

考虑到vrp的一般优化目标是获得最短的路由长度，包含相邻节点的局部邻域在节点间移动的决策中起着重要作用。同时，局部拓扑特征在节点分布和规模之间具有很大的可转移潜力。为了利用本地信息的属性，本文提出了一种新的本地策略，该策略将状态和操作空间限制在少数本地邻居节点上。此外，本文将本地策略与称为全局策略的典型构建策略（例如POMO）集成在一起，全局策略从完整VRP实例的全局信息中学习，形成一个集成策略。对局部策略和全局策略进行联合训练，实现协同互补，达到良好的泛化效果。

- 问题：TSP 和 CVRP
- 训练：均匀分布 小规模
- 测试：TSBLIB，CVRPLIB
- 实证结果表明，集成策略显著提高了交叉分布和跨尺度的泛化性能，并且在大多数情况下优于现有的构建方法（如 BQ 和 LEHD ）
- 所提出的方法甚至在具有数千个节点的现实问题上表现良好，而大多数构造方法很难直接解决此类现实问题。
- 消融研究也验证了可转移局部策略对更好的泛化的关键作用。

## 相关工作

- VRP的定义及建模
- 神经CO方法
  - 自回归的方式构造序列解
  - MDP
  - AM POMO的计算方法
- 泛化问题
  - 缓解方法：分治、元学习、CO问题的对称性、重解码器

## 方法

Ensemble of Local and Global policies (ELG)，包含两个基本策略：一个主要的全局策略和一个辅助的局部策略。

主全局策略可以是任何流行的神经组合优化模型，它从一个完整的VRP实例的全局信息中学习。具体来说，本文选择POMO作为全局策略。局部策略解的构造过程与全局策略解的构造过程类似，但状态和动作空间不同，其重点是当前节点的局部邻域。

![image-20241022114443922]({{site.url}}/img/2024-10-22-Towards_Generalizable_Neural_Solvers_for_Vehicle_Routing_Problems_via_Ensemble_with_Transferrable_Local_Policy/image-20241022114443922.png)

### 可转让局部策略

在求解vrp时，一个常见的观察结果是，最优动作（即下一个节点）通常包含在当前节点的一个小局部邻域中，并且这些局部邻域的模式具有跨各种节点分布和尺度可转移的潜力。受这些事实的启发，本文提出设计一个局部MDP公式，将状态和动作空间限制在局部邻域，以获得更好的泛化性能。

- 状态：状态空间被缩减到一个小的局部邻域 $\mathcal{N}_K\left(c_t\right)$，该邻域包含当前节点 $c_t$ 的 $K$ 个最近的有效邻居节点。有效节点是指未被访问且满足约束的节点，例如，在车辆路径问题（CVRP）中的容量约束。直观上，局部状态表示策略在接下来的几步中需要解决的一个子问题。通过学习解决局部子问题，策略捕捉到了车辆路径问题的更多内在特征，这些特征可以在不同的问题实例中迁移。为了更好地表示局部状态的特征，我们利用以当前节点 $c_t$ 为中心的极坐标 $\left(\rho_i, \theta_i\right)$ 来指示邻居节点的位置，其中 $i$ 是节点 $n_i \in \mathcal{N}_K\left(c_t\right)$ 的索引。极坐标直接提供到 $c_t$ 的相对距离（即边缘成本），我们可以通过 $\tilde{\rho}_i=\rho_i / \max \left\{\rho_i \mid n_i \in \mathcal{N}_K\left(c_t\right)\right\}$ 将所有的 $\rho_i$ 归一化到 $[0,1]$。因此，所有邻居节点都位于单位球内。这种局部拓扑特征对节点分布和问题规模的变化不敏感。CVRP 的状态还包含节点需求 $\left\{\tilde{d}_i \mid n_i \in \mathcal{N}_K\left(c_t\right)\right\}$，按剩余容量 $Q_{\text{remain}}$ 归一化，即 $\tilde{d}_i=d_i / Q_{\text{remain}}$。因此，$Q_{\text{remain}}$ 不再需要包含在状态中。

- 动作：局部策略输出用于选择下一个要访问的邻居节点的分数 $\boldsymbol{u}_{\text{local}}$，其中节点 $n_i$ 的分数可以表示为

  $$
  u_{\text{local}}^i=\left\{\begin{array}{c}
  \left(g_{\boldsymbol{\theta}}\left(s_t\right)\right)_i, \text{ 如果 } n_i \in \mathcal{N}_K\left(c_t\right), \\
  0, \text{ 否则 }
  \end{array}\right.
  $$

  其中 $\boldsymbol{s}_t=\left\{\left[\tilde{\rho}_i, \theta_i, \tilde{d}_i\right] \mid n_i \in \mathcal{N}_K\left(c_t\right)\right\}$ 适用于 CVRP（注意，需求 $\tilde{d}_i$ 在旅行商问题（TSP）中被移除），$g_{\boldsymbol{\theta}}$ 是一个参数化的神经网络。

本文提出的局部策略的优点可以概括为：

1. 局部模式对节点分布和问题规模的变化不敏感，具有可转移性；
2. 局部MDP提供了良好的归纳偏差，使得策略可以集中在可能更有前途的局部邻居节点上。

### 神经架构

POMO

### 全局策略和局部策略集成

在整合之前，为了更好地泛化，本文调整了原来的全局策略，增加了距离惩罚，如下所示。

#### 全局策略的标准化距离惩罚

考虑到大多数最优动作都包含在局部邻居节点中，利用归一化的距离对全局策略进行惩罚。距离惩罚鼓励策略偏向于选择附近的节点，并对选择远程节点保持谨慎，这在实际应用中有助于泛化。与之前直接将距离值作为偏置的方法不同，本文提出通过 $\mathcal{N}_K\left(c_t\right)$ 中最大的 $\rho_i$ 将距离 $\rho_i$ 归一化到 $[0,1]$，并对非邻居节点添加一个固定惩罚 $\xi (\xi \geq 1)$，即：

$$
\tilde{u}_{\text {global }}^i=\left\{\begin{aligned}
u_{\text {global }}^i-\frac{\rho_i}{\max \left\{\rho_i \mid n_i \in \mathcal{N}_K\left(c_t\right)\right\}}, & \text { if } n_i \in \mathcal{N}_K\left(c_t\right), \\
u_{\text {global }}^i-\xi, & \text { otherwise }
\end{aligned}\right.
$$

$\mathcal{N}_K\left(c_t\right)$ 包含当前节点 $c_t$ 的 $K$ 个最近有效邻居节点，这为归一化提供了良好的视角，使距离惩罚更具可扩展性。

为了整合全局策略和局部策略，首先将两个基础策略计算的动作分数相加，即 $\boldsymbol{u}_{\text{ens}} = \tilde{\boldsymbol{u}}_{\text{global}} + \boldsymbol{u}_{\text{local}}$。之后，集成策略的动作概率 $\pi_{\text{ens}}$ 通过以下公式计算：

$$
\begin{aligned}
u_{\text{masked}}^i & = \left\{\begin{aligned}
C \cdot \tanh \left(\tilde{u}_{\text{global}}^i + u_{\text{local}}^i\right), & \text{如果节点 } n_i \text{ 有效,} \\
-\infty, & \text{否则},
\end{aligned}\right. \\
\boldsymbol{\pi}_{\text{ens}} & = \operatorname{softmax}\left(\boldsymbol{u}_{\text{masked}}\right) .
\end{aligned}
$$

本文并不是独立训练每个基础策略，而是使用联合训练方法直接优化 $\pi_{\text{ens}}$，以鼓励全局策略和局部策略协同工作，从而有效结合它们的优势。

#### 训练

在实践中，首先对全局策略进行距离惩罚的预训练，持续 $T_1$ 轮，因为全局策略的状态和动作空间比局部策略更复杂。在联合训练阶段，使用策略梯度方法直接训练集成策略 $\pi_{\text{ens}}$，持续 $T_2$ 轮，该策略包括来自全局策略的可训练参数 $\tilde{\boldsymbol{\theta}}$ 和来自局部策略的 $\boldsymbol{\theta}$。按照 POMO，从不同的起始节点进行多次回放，在一次前馈中获得多条轨迹，并利用 REINFORCE 算法与共享基线估计预期回报 $J$ 的梯度。具体而言，多次回放的平均奖励作为 REINFORCE 基线，梯度 $\nabla_{\tilde{\boldsymbol{\theta}}, \boldsymbol{\theta}} J(\tilde{\boldsymbol{\theta}}, \boldsymbol{\theta})$ 通过以下公式估计：

$$
\frac{1}{N \cdot B} \sum_{i=1}^B \sum_{j=1}^N\left(R_{i, j}-\frac{1}{N} \sum_{j=1}^N R_{i, j}\right) \nabla_{\tilde{\boldsymbol{\theta}}, \boldsymbol{\theta}} \log \boldsymbol{\pi}_{\mathrm{ens}}\left(\boldsymbol{\tau}_{i, j}\right),
$$

其中 $N$ 是轨迹的数量，等于节点的数量，$B$ 是批量大小，$R_{i, j}$ 是第 $i$ 个实例上第 $j$ 条轨迹 $\boldsymbol{\tau}_{i, j}$ 的奖励。

## 实验

### 实验设置

- baseline
  - 非学习启发式：LKH3和HGS
  - POMO-base方法：POMO和sym-POMO
  - cross-distribution方法：AMDKD和Omni-POMO
  - cross-scale方法：Pointerformer、LEHD、BQ、TAM
  - diffusion方法：DIFUSCO、T2TCO
- 数据集
  - TSPLIB、CVRPLIB Set X & XXL：这些基准涵盖了具有复杂和未知分布以及变化和大尺度的各种问题实例，可用于评估更现实场景中的交叉分布和跨尺度泛化。
  - cross-distribution、cross-scale：使用具有复杂合成分布（包括聚类、聚类混合、爆炸和旋转）的小规模实例。
  - 均匀小尺度实例
- 结果标准：与数据集上报告的最佳解的平均Gap
- 训练：所有的学习方法都在规模为100的均匀分布上训练

### 实验结果

<img src="{{site.url}}/img/2024-10-22-Towards_Generalizable_Neural_Solvers_for_Vehicle_Routing_Problems_via_Ensemble_with_Transferrable_Local_Policy/image-20241022153834566.png" alt="image-20241022153834566" style="zoom:50%;" />

四个现实世界的实例

<img src="{{site.url}}/img/2024-10-22-Towards_Generalizable_Neural_Solvers_for_Vehicle_Routing_Problems_via_Ensemble_with_Transferrable_Local_Policy/image-20241022153920377.png" alt="image-20241022153920377" style="zoom:50%;" />

不同分布的结果

<img src="{{site.url}}/img/2024-10-22-Towards_Generalizable_Neural_Solvers_for_Vehicle_Routing_Problems_via_Ensemble_with_Transferrable_Local_Policy/image-20241022154206819.png" alt="image-20241022154206819" style="zoom:50%;" />

训练数据同规模同分布测试

<img src="{{site.url}}/img/2024-10-22-Towards_Generalizable_Neural_Solvers_for_Vehicle_Routing_Problems_via_Ensemble_with_Transferrable_Local_Policy/image-20241022154445249.png" alt="image-20241022154445249" style="zoom:50%;" />

### 消融实验

对与局部策略相关的三个组成部分进行了消融研究：归一化距离惩罚、位置编码和极坐标。

两个新超参数：固定惩罚 $ξ$ 和局部邻域大小 $K$ 的影响。

（没看附录）

## 总结

所提出的方法的一个限制是，它需要花费更多的时间来解决非常大规模的实例比分治方法。

在未来的工作中，将尝试进一步改善延迟，并将所提出的方法扩展到更多的CO问题，例如PDP。





