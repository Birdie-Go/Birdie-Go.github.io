---
layout:     post
title:      UDC A Unified Neural Divide-and-Conquer Framework for Large-Scale Combinatorial Optimization Problems
subtitle:   NIPS2024 大规模 多COP框架 分治策略
date:       2024/11/01
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - NIPS
---

UDC: A Unified Neural Divide-and-Conquer Framework for Large-Scale Combinatorial Optimization Problems

南方科技大学、华为诺亚方舟实验室

王振坤课题组

NIPS 2024

开源：https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/UDC-Large-scale-CO-master



## 摘要

单阶段神经组合优化求解器在不需要专家知识的情况下，对各种小尺度组合优化（CO）问题取得了接近最优的结果。然而，当应用于大规模CO问题时，这些求解器表现出显著的性能下降。近年来，由分而治之策略驱动的两阶段神经方法在解决大规模CO问题上显示出高效率。然而，这些方法的性能在划分或求解过程中高度依赖于特定于问题的启发式，这限制了它们对一般CO问题的适用性。此外，这些方法采用单独的训练方案，忽略了分割和求解策略之间的相互依赖性，经常导致次优解。为了解决这些缺点，本文开发了一个统一的神经分治框架（即UDC），用于解决一般的大规模CO问题。UDC提供了一种划分-求解-重聚（DCR）训练方法来消除次优划分策略的负面影响。采用高效的图神经网络（GNN）进行全局实例划分，采用定长子路径求解器求解划分子问题，所提出的UDC框架具有广泛的适用性，在10个具有代表性的大规模CO问题中取得了优异的性能。

## 介绍

现阶段大规模NCO主要方法：局部策略、分治（Generalize learned heuristics to solve large-scale vehicle routing problems in real-time、GLOP）

BQ-NCO 和 LEHD 开发了子路径构建过程，并使用带有重型解码器的模型来学习该过程。但需要重量级监督学习。

ELG、ICAM、DAR 整合了辅助信息来知道基于 RL 的但阶段构造求解器学习。但辅助信息需要针对问题设计，self-attention 复杂度也很大。

分治分为两个阶段：划分阶段、子问题求解阶段，包括 TAM、H-TSP、GLOP。

神经分治不足：依赖特定问题的启发式，不利于推广到不同问题。且两个阶段独立，忽略两个阶段之间的相互依赖关系。

![image-20241101163935467]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101163935467.png)

本文提出了新的强化学习框架 Divide-Conquer-Reunion (DCR)，并提出了 neural divide-and-conquer framework (UDC) 用于求解大规模问题。

实验结果表明，在不依赖任何启发式设计的情况下，UDC在效率和适用性方面都明显优于现有的大规模NCO求解器。

贡献：

1. 提出了一种新的DCR方法，通过减轻次优划分策略的负面影响来增强训练。
2. 将DCR运用到训练中，所提出的UDC实现了统一的训练方案，并具有显著的性能优势。
3. UDC具有广泛的适用性，可以应用于具有类似设置的一般CO问题。

## 相关工作：神经分治

- 神经分治算法一般范式
- 神经构造求解器一般范式
- 基于热图的神经求解器一般范式

## 方法论：unified 分治

![image-20241101170011218]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101170011218.png)

1. 构造一个稀疏图表示实例
2. 使用基于热图的求解器，并以 Anisotropic Graph Neural Network (AGNN) 作为划分策略生成初始解
3. 在初始解的基础上将原始实例分为多个子问题，并进行必要的约束
4. 采用神经构造求解器求解（针对 MIS（最大独立集） 的 AGNN，针对 VRP 的 POMO 和 ICAM）

### 分治策略：基于热图的求解器

基于热图的求解器相比构造性求解器，需要更少的时间和空间消耗，因此它们更适合于学习大规模实例上的全局分割策略。分割阶段首先为原始CO实例 $\mathcal{G}$ 构建一个稀疏图 $\mathcal{G}\_D=\lbrace \mathbb{V}, \mathbb{E}\rbrace$ （即，在TSP中链接K最近邻（KNN），或在 $\mathcal{G}$ 中的原始边对于MIS）。然后，使用参数为 $\phi$ 的各向异性图神经网络（AGNN）来生成热图 $\mathcal{H}$。对于 $N$ 节点的VRPs，热图 $\mathcal{H} \in \mathbb{R}^{N \times N}$ 以及基于策略 $\pi\_d$ 生成的 $\tau$ 长度的初始解 $\boldsymbol{x}\_0=\left(x\_{0,1}, \ldots, x\_{0, \tau}\right)$ 如下：

$$
\pi_d\left(\boldsymbol{x}_0 \mid \mathcal{G}_D, \Omega, \phi\right)=\left\lbrace \begin{array}{ll}
p\left(\mathcal{H} \mid \mathcal{G}_D, \Omega, \phi\right) p\left(x_{0,1}\right) \prod_{t=2}^\tau \frac{\exp \left(\mathcal{H}_{x_{0, t-1}, x_{0, t}}\right)}{\sum_{i=t}^N \exp \left(\mathcal{H}_{x_{0, t-1}, x_{0, i}}\right)}, & \text { if } \boldsymbol{x}_0 \in \Omega \\
0, & \text { otherwise }
\end{array} .\right.
$$

### 求解阶段：子问题准备

求解阶段首先生成待处理的子问题及其特定约束。对于VRPs，子VRPs中的节点和子VRPs的约束是基于初始解 $x\_0$ 中的连续子序列构建的。在生成 $n$ 节点子问题时，UDC根据 $\boldsymbol{x}^0$ 将原始的 $N$ 节点实例 $\mathcal{G}$ 分成 $\left\lfloor\frac{N}{n}\right\rfloor$ 个子问题 $\left\lbrace \mathcal{G}\_1, \ldots, \mathcal{G}\_{\left\lfloor\frac{N}{n}\right\rfloor}\right\rbrace$ ，暂时排除少于 $n$ 节点的子问题。子问题的约束 $\left\lbrace \Omega\_1, \ldots, \Omega\_{\left\lfloor\frac{N}{n}\right\rfloor}\right\rbrace$ 不仅包括问题特定的约束（例如，子TSPs中没有自环），还包括额外的约束以确保将子问题解决方案整合到原始解决方案后合并解决方案的合法性（例如，保持子VRPs中的第一和最后一个节点）。还需要标准化子问题的坐标和一些数值约束，以增强待处理子问题实例的同质性。由于需要处理不同的约束，不同CO问题的子问题准备过程各不相同。

### 求解阶段：神经构造

在子问题准备之后，求解策略被用来为这些实例 $\left\lbrace \mathcal{G}\_1, \ldots, \mathcal{G}\_{\left\lfloor\frac{N}{n}\right\rfloor}\right\rbrace$ 生成解决方案。我们使用参数为 $\theta$ 的构造性求解器来处理大多数涉及的子CO问题。它们的子解决方案 $s\_k=\left(s\_{k, 1}, \ldots, s\_{k, n}\right), k \in\left\lbrace 1, \ldots,\left\lfloor\frac{N}{n}\right\rfloor\right\rbrace$ 是从求解策略 $\pi\_c$ 中采样的，如下所示：

$$
\pi_c\left(\boldsymbol{s}_k \mid \mathcal{G}_k, \Omega_k, \theta\right)=\left\lbrace \begin{array}{lc}
\prod_{t=1}^n p\left(s_{k, t} \mid s_{k, 1: t-1} \mathcal{G}_k, \Omega_k, \theta\right), & \text { if } \boldsymbol{s}_k \in \Omega_k \\
0, & \text { otherwise }
\end{array}\right.
$$

其中 $s\_{k, 1: t-1}$ 表示在选择 $s\_{k, t}$ 之前的部分解决方案。最后，在求解阶段的最后一步，改进目标函数的子解决方案将替换 $\boldsymbol{x}\_0$ 中的原始解决方案片段，合并后的解决方案变为 $\boldsymbol{x}\_1$。值得注意的是，求解阶段可以在新的合并解决方案上重复执行，以获得更好的解决方案质量，记 $r$ 次求解阶段后的解决方案为 $\boldsymbol{x}\_r$。

### 训练模型：分-治-重新联合

分和治两阶段都可以建模为 MDP。治的奖励是子问题的目标函数，分的奖励根据最后合并的目标函数计算。

现有的神经分治方法无法与强化学习同时训练两种策略，因此，通常采用单独的训练方案，但这破坏了分治策略的协同性。

![image-20241101194309075]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101194309075.png)

DCR设计了一个额外的再合并步骤，将原始两个相邻子问题之间的连接视为一个新的子问题，并在 $\boldsymbol{x}\_1$ 上进行另一个求解阶段，采用新的子问题分解。再合并步骤可以表述为将原始起点 $x\_{1,0}$ 沿着解决方案 $\boldsymbol{x}\_{1,0}$ 滚动 $l$ 次（即，到 $x\_{1, l}$ ）并再次求解。为了确保对相邻两个子问题的同等关注，我们在训练所有启用DCR的UDC模型时设置 $l=\frac{n}{2}$ 。DCR为划分策略的奖励提供了更好的估计，从而提高了统一训练中的稳定性和收敛速度。

UDC在训练划分策略（损失 $\mathcal{L}\_d$ ）和求解策略（求解步骤的损失 $\mathcal{L}\_{c 1}$ 和再合并步骤的损失 $\mathcal{L}\_{c 2}$ ）时采用REINFORCE算法。划分策略训练的基线是 $\alpha$ 个初始解决方案 $\boldsymbol{x}\_0^i, i \in\lbrace 1, \ldots, \alpha\rbrace$ 的平均奖励，求解策略的基线是计算在 $\beta$ 个抽样子解决方案上。单个实例 $\mathcal{G}$ 上的损失函数梯度计算如下：

$$
\begin{aligned}
\nabla \mathcal{L}_d(\mathcal{G}) & =\frac{1}{\alpha} \sum_{i=1}^\alpha\left[\left(f\left(\boldsymbol{x}_2^i, \mathcal{G}\right)-\frac{1}{\alpha} \sum_{j=1}^\alpha f\left(\boldsymbol{x}_2^j, \mathcal{G}\right)\right) \nabla \log \pi_d\left(\boldsymbol{x}_2^i \mid \mathcal{G}_D, \Omega, \phi\right)\right], \\
\nabla \mathcal{L}_{c 1}(\mathcal{G}) & =\frac{1}{\alpha \beta\left\lfloor\frac{N}{n}\right\rfloor} \sum_{c=1}^{\alpha\left\lfloor\frac{N}{n}\right\rfloor} \sum_{i=1}^\beta\left[\left(f^{\prime}\left(\boldsymbol{s}_c^{1, i}, \mathcal{G}_c^0\right)-\frac{1}{\beta} \sum_{j=1}^\beta f^{\prime}\left(\boldsymbol{s}_c^{1, j}, \mathcal{G}_c^0\right)\right) \nabla \log \pi_c\left(\boldsymbol{s}_c^{1, j} \mid \mathcal{G}_c^0, \Omega_c^0, \theta\right)\right], \\
\nabla \mathcal{L}_{c 2}(\mathcal{G}) & =\frac{1}{\alpha \beta\left\lfloor\frac{N}{n}\right\rfloor} \sum_{c=1}^{\alpha\left\lfloor\frac{N}{n}\right\rfloor} \sum_{i=1}^\beta\left[\left(f^{\prime}\left(\boldsymbol{s}_c^{2, i}, \mathcal{G}_c^1\right)-\frac{1}{\beta} \sum_{j=1}^\beta f^{\prime}\left(\boldsymbol{s}_c^{2, j}, \mathcal{G}_c^1\right)\right) \nabla \log \pi_c\left(\boldsymbol{s}_c^{2, j} \mid \mathcal{G}_c^1, \Omega_c^1, \theta\right)\right],
\end{aligned}
$$

其中 $\left\lbrace \boldsymbol{x}\_2^1, \ldots, \boldsymbol{x}\_2^\alpha\right\rbrace$ 表示 $\alpha$ 个抽样解决方案。基于 $\left\lbrace \boldsymbol{x}\_0^1, \ldots, \boldsymbol{x}\_0^\alpha\right\rbrace$ 在第一求解阶段生成的 $\alpha\left\lfloor\frac{N}{n}\right\rfloor$ 个子问题 $\mathcal{G}\_c^0, c \in$ $\left\lbrace 1, \ldots,\left\lfloor\frac{N}{n}\right\rfloor, \ldots, \alpha\left\lfloor\frac{N}{n}\right\rfloor\right\rbrace$，约束条件为 $\Omega\_c^0$ 。 $\alpha\left\lfloor\frac{N}{n}\right\rfloor$ 可以被视为子问题的批量大小，$\mathcal{G}\_c^1, \Omega\_c^1, c \in\left\lbrace 1, \ldots, \alpha\left\lfloor\frac{N}{n}\right\rfloor\right\rbrace$ 是第二求解阶段的子问题和约束条件。子问题 $\mathcal{G}\_c^0, \mathcal{G}\_c^1, c \in\left\lbrace 1, \ldots, \alpha\left\lfloor\frac{N}{n}\right\rfloor\right\rbrace$ 的 $\beta$ 个抽样子解决方案分别记为 $\left\lbrace \boldsymbol{s}\_c^{1, i}, \ldots, \boldsymbol{s}\_c^{1, \beta}\right\rbrace,\left\lbrace \boldsymbol{s}\_c^{2, i}, \ldots, \boldsymbol{s}\_c^{2, \beta}\right\rbrace$。

### 应用：适用于一般CO问题

应用于满足以下三个条件的一般离线CO问题：

- 目标函数仅包含可分解的聚合函数（即不包含Rank或Top-k等函数）。
- 初始解和子解的合法性可以通过可行性掩码来保证。
- 被划分子问题的解并不总是唯一的。

对于第二个条件，对于无法通过自回归解决过程（例如TSPTW）保证解决方案合法的复杂CO问题（例如TSPTW），UDC是无效的。

对于第三种情况，对于某些CO问题，如密集图上的MIS问题或作业调度问题，子问题的解已经唯一确定，因此求解阶段失效。

## 实验

10个组合优化问题（包括TSP、CVRP、KP、MIS、ATSP、Orienteering problem （OP）、PCTSP、Stochastic PCTSP （SPCTSP）、Open VRP（OVRP））

可直接在 TSP500 和 TSP1000 上训练。在 V100 上训练 10 天。

baseline

- 经典求解器：LKH、EA4OP、ORTools
- 监督学习：BQ、LEHD
- 强化学习：AM、POMO、ELG、ICAM、MDAM
- 热图：DIMES、DIFUSCO
- 神经分治：GLOP、TAM、H-TSP

![image-20241101201227189]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101201227189.png)

标准数据集

![image-20241101201311400]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101201311400.png)

## 讨论

### 分别训练和联合训练

![image-20241101201723557]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101201723557.png)

联合训练是有效的

### 消融实验

求解策略

<img src="{{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101201751318.png" alt="image-20241101201751318" style="zoom:50%;" />

## 总结

未来工作：

- UDC的损失函数设计还有提升空间。
- 将UDC的适用性扩展到上述不满足三点要求的CO问题

## 附录

### 相关工作

各种方法详细介绍

- 神经CO
- 构造求解器
- 热图求解器
- 神经分治

### 细节定义

- 大规模CO的定义
- 问题和子问题的定义：TSP、CVRP、OP、PCTSP、SPCTSP、KP、OVRP、ATSP、MIS、Min-max mTSP

### 方法细节

- 分治阶段：图构建

  - 稀疏图：对于VRP，用KNN连边

- 分治策略：AGNN

- 求解策略：子问题准备

  - 保持以确定的解和不变的部分（如VRP的起点）
  - 将原始约束分配到子问题中（如CVRP的容量）
  - 考虑原CO问题的约束度

- 求解阶段：标准化

- 求解阶段：求解策略

- 伪代码

  ![image-20241101202935766]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101202935766.png)

  ![image-20241101202959694]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101202959694.png)

- MDP：分治阶段MDP和求解阶段MDP

- 时空复杂度分析

  ![image-20241101203037061]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203037061.png)

### 实验

- 超参数设置
- 其他问题的实验

### 消融实验

- $r$ 的收敛速度

  ![image-20241101203141795]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203141795.png)

- $\alpha$，初始解抽样

  ![image-20241101203210502]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203210502.png)

- 不同初始解

  ![image-20241101203243983]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203243983.png)
  
- 其他组件
  
  ![image-20241101203306972]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203306972.png)
  
- 不使用 DCR

  ![image-20241101203340048]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203340048.png)

### 可视化

![image-20241101203358779]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203358779.png)

![image-20241101203407153]({{site.url}}/img/2024-11-01-UDC_A_Unified_Neural_Divide-and-Conquer_Framework_for_Large-Scale_Combinatorial_Optimization_Problems/image-20241101203407153.png)

###   baseline 和 license

略

（完结，附件太多了，一共45页）































