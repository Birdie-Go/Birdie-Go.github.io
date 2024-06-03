---
layout:     post
title:      Let the Flows Tell Solving Graph Combinatorial Problems with GFlowNets
subtitle:   NIPS23 GFlowNets
date:       2024/1/23
author:     Birdie
header-img: img/post_header_sr.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - NIPS
---

NIPS23 Let the Flows Tell Solving Graph Combinatorial Problems with GFlowNets

用GFlowNets求解图组合优化问题

来自Mila实验室和Google DeepMind

开源：https://github.com/zdhNarsil/GFlowNet-CombOpt

（没研究过相关内容，没看懂）

## 摘要

组合优化(CO)问题通常是np困难的，因此无法使用精确的算法，这使得它们成为应用机器学习方法的诱人领域。这些问题中的高度结构化约束可能会阻碍优化或直接在解空间中采样。另一方面，GFlowNets最近作为一种强大的机器出现，可以有效地从复合非归一化密度中依次取样，并且有可能在CO中平复这种解决方案搜索过程，并生成多种候选解决方案。在本文中，我们针对不同的组合问题设计了马尔可夫决策过程(mdp)，并提出了训练条件GFlowNets从解空间中采样的方法。有效的培训技术也被开发出来，有利于长期的信用分配。通过对各种不同CO任务的综合和现实数据的大量实验，我们证明GFlowNet策略可以有效地找到高质量的解决方案。我们的实现是开源的，网址是https://github.com/zdhNarsil/GFlowNet-CombOpt。



## 介绍

- 监督学习（SL）：需要数值求解器进行昂贵的计算来生成标签
- 无监督学习（UL）
  - 概率方法，生成热图进行解码：快速推理但效果一般
  - 强化学习，MDP：问题有多个最优解，普通的强化学习有确定性策略不能促进解的多样性；性能很大程度取决奖励函数的设计，并以来学习价值函数和粗略的每步奖励。

目前的改进都是针对特定问题的，作者使用了一个原则性框架，生成流网络（GFlowNets，2021年提出的）。GFlowNet是一种新颖的决策框架，用于学习随机策略，以与给定最终奖励成比例的概率对复合对象进行采样，适用于解只与生成轨迹的终端状态相关的问题。

贡献：

- 针对四个不同的CO为GFlowNet设计了一个问题特定的MDP
- 提出了高效的GFlowNet学习方法，以实现具有长轨迹的GFlowNet代理的快速credit分配
- 实验验证GFlowNets的优势



## 前言

### GFlowNets

生成流网络是一种变分（variational）推理算法，它将目标概率分布的采样视为一个连续的决策过程。

假设状态集为 $\mathcal{S}$ 以及动作集合 $\mathcal{A}\subseteq\mathcal{S}\times\mathcal{S}$。一个确定性的MDP，其初始状态为 $s_0$，终止状态没有输出动作，表示为 $\mathcal{X}$。一个完整的轨迹是一系列的状态 $\tau=(s_0\rightarrow s_1\rightarrow\cdots\rightarrow s_n)$，其中 $s_n\in\mathcal{X}$，且 $\forall i(s_i,s_{i+1})\in\mathcal{A}$。MDP上的一个策略是分布 $P_F(s'\mid s)$ 的一个选择，其中 $s\in\mathcal{S}\setminus\mathcal{X}$ 且 $s'$ 是从 $s$ 一步得到的。一个策略经过一个完整的轨迹推导出一个分布

$$
P_{F}(\mathbf{s}_{0}\to\mathbf{s}_{1}\to\cdots\to\mathbf{s}_{n})=\prod_{i=0}^{n-1}P_{F}(\mathbf{s}_{i+1}\mid\mathbf{s}_{i})
$$

完整轨迹在终止状态的边缘分布记作 $P_F^{\top}$。这是一个在 $\mathcal{X}$ 上的分布，通常难以精确计算，

$$
P_{F}^{\top}(\mathbf{x})=\sum_{\tau\to\mathbf{x}}P_{F}(\tau)
$$

奖励函数是一个映射 $\mathcal{X}\rightarrow\mathbb{R}_{>0}$，可以理解为终止状态集合上的非归一化概率质量，通常表示为 $R(\mathbf{x})=\exp(-\mathcal{E}(\mathbf{x})/T)$，其中 $\mathcal{E}:\mathcal{X}\rightarrow\mathbb{R}$ 是一个能量函数且 $T>0$ 是一个温度参数。GFlowNet 近似解决的学习问题是拟合一个参数 $P_F(s'\mid s)$，使得诱导分布 $P_F^{\top}$ 与奖励函数成正比，

$$
P_{F}^{\top}(\mathbf{x})\propto R(\mathbf{x})=\exp(-\mathcal{E}(\mathbf{x})/T).
$$

将策略 $P_F(s'\mid s)$ 参数化为一个神经网络，以参数 $\theta$ 作为输入，并产生每个可能的后续状态 $s'$ 的转移概率。在给定 $P_F$ 的情况下，计算 $P_F^{\top}$ 是很困难的，而且奖励函数中的归一化常数是位置的。

学习算法通过在优化中引入辅助对象来克服这些困难。

#### Detailed balance（DB）

DB目标是学习两个对象以及参数转发策略 $P_F(s'\mid s;\theta)$

- backward policy：$P_F(s\mid s';\theta)$ 表示任何非初始状态的前驱
- state flow：$F(\cdot;\theta):\mathcal{S}\rightarrow\mathbb{R}_{>0}$

DB loss对于打死你个转移 $s\rightarrow s'$ 定义为

$$
\ell_{\mathrm{DB}}(\mathbf{s},\mathbf{s'};\boldsymbol{\theta})=\left(\log\frac{F(\mathbf{s};\boldsymbol{\theta})P_F(\mathbf{s'}|\mathbf{s};\boldsymbol{\theta})}{F(\mathbf{s'};\boldsymbol{\theta})P_B(\mathbf{s}|\mathbf{s'};\boldsymbol{\theta})}\right)^2.
$$

DB训练理论表明，如果对于任意的转移 $s\rightarrow s'$ 都有 $\ell_{\mathrm{DB}}(\mathbf{s},\mathbf{s'};\boldsymbol{\theta})=0$，那么策略 $P_F$ 会满足上述诱导分布 $P_F^{\top}$ 与奖励函数成正比。

在本文中的问题，表现最好的损失相当于具有 $\log F(s)$ 特定参数化的DB，改参数化通过将log-state flow表示为部分累计的负能量的加性校正来引导学习。

#### Trajectory balance（TB）

TB目标除了动作策略 $P_F$ 之外，还学习了一个后向策略 $P_B$ 和一个单一标量 $Z_\theta$，他是DB参数化中与初始状态流 $F(s_0)$ 对应的分配函数的估计量。完整轨迹的TB损失为

$$
\ell_\mathrm{TB}(\tau;\boldsymbol{\theta})=\left(\log\frac{Z_{\boldsymbol{\theta}}\prod_{i=0}^{n-1}P_{F}(\mathbf{s}_{i+1}|\mathbf{s}_{i};\boldsymbol{\theta})}{R(\mathbf{x})\prod_{i=0}^{n-1}P_{B}(\mathbf{s}_{i}|\mathbf{s}_{i+1};\boldsymbol{\theta})}\right)^2.
$$

TB训练理论表明，如果对于所有的完整轨迹都有 $\ell_\mathrm{TB}(\tau;\boldsymbol{\theta})=0$，那么策略 $P_F$ 会满足上述诱导分布 $P_F^{\top}$ 与奖励函数成正比。进一步来说，$\hat{Z}=\sum_{\mathbf{x}\in\mathcal{X}}R(\mathbf{x})$。

进一步来说，策略和流通常在log域中输出，即神经网络预测分布 $P_F(\cdot\mid s),P_B(\cdot\mid s')$ ，log-flows $F(s)$ 和 $\log Z$。

#### 训练策略和探索

DB和TB的损失取决于个体转移或者轨迹，但是如何选择使其最小化的转移或者轨迹的问题仍未解决。一种常见的方法是以非策略的方式进行训练，即rollout策略 $\tau\sim P_F(\tau;\theta)$，并对 $\ell_\mathrm{TB}(\tau;\boldsymbol{\theta})$ 或者 转移 $s_i\rightarrow s_{i+1}$ 上的 $\ell_{\mathrm{DB}}(\mathbf{s}_i,\mathbf{s}_{i+1};\boldsymbol{\theta})$ 进行梯度下降。在这种情况下，DB和TB与变分（ELBO最大化）的目标密切相关。

也可以通过探索性行为策略，例如，从均匀分布的 $P_F$ 中采样 $\tau$，比如 $\epsilon$ -greedy。

与RL中的策略梯度方法不同，GFlowNet目标不需要通过产生τ的采样过程进行微分。从非策略轨迹中稳定学习的能力是GFlowNets相对于分层变分模型的一个关键优势。

#### 条件GFlowNets

GFlowNet中的MDP和奖励函数可以依赖于一些条件信息。例如，在研究的任务中，GFlowNet策略顺序地在图 $g$ 上构造CO问题的解，并且允许的动作集依赖于 $g$ 。我们训练的条件GFlowNets通过在不同的 $g$ 之间共享策略模型来实现分摊，从而实现对训练中未见的 $g$ 的泛化。



## 方法

已经完全看不懂了，前置知识匮乏，针对的CO问题也没研究过，建议直接看原文。