---
layout:     post
title:      Enhancing LLM Safety via Constrained Direct Preference Optimization
subtitle:   ICLR2024 受约束的直接偏好优化
date:       2025/04/09
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - ICLR
    - 大模型
---

Enhancing LLM Safety via Constrained Direct Preference Optimization

杜兰大学

ICLR 2024



## 摘要

随着大型语言模型（LLMs）能力的迅速提升，迫切需要将 AI 系统与多样化的人类偏好对齐，以同时增强其有用性和安全性。尽管这些目标往往存在冲突，但实现对齐是至关重要的。为了应对这一挑战，一种有前景的方法是在微调阶段通过受约束的强化学习从人类反馈（RLHF）框架来强制执行安全性约束。然而，这种方法计算成本高昂且通常不稳定。在本文中，提出了一种名为 **受约束的直接偏好优化（Constrained DPO, C-DPO）** 的新方法，它是最近提出的直接偏好优化（DPO）方法的扩展，用于微调大型语言模型。该方法既高效又轻量，通过结合对偶梯度下降和 DPO，在不使用强化学习的情况下，找到了有用性和无害性之间的近似最优权衡。实验表明，我们的方法为 LLMs 提供了缺失的安全性保障，同时在相同的安全性约束下，相较于最近提出的基于安全 RLHF 的方法，我们的方法能够显著提高奖励。

## 介绍

大型语言模型（LLMs）在聊天补全、指令遵循、编程、问题解决和决策制定等任务中展现出了卓越的能力。然而，它们也存在各种弱点和漏洞，这可能会阻碍它们在安全和安全关键领域的应用。为了使这些模型更符合人类偏好，人们采用了监督微调（SFT）以及基于人类反馈（RLHF）或人工智能反馈（RLAIF）的强化学习等技术。然而，这些技术在面对经过精心设计的对抗性输入时，无法提供强大的防御。这种局限性源于LLM训练中固有的冲突目标，例如有用性和无害性，这些目标很难通过单一的奖励或偏好模型来平衡。

**一个有前景的提升安全性的方向是将奖励和安全性目标解耦，并在安全性约束下对LLM进行微调，以优化预期奖励，其中目标和约束使用来自人类（或人工智能）反馈的不同数据集进行建模。**通过施加安全性约束，这种方法可以在不降低模型效用的情况下，潜在地导致更安全的模型。此外，它自然地融入了最近广泛研究的安全强化学习框架。然而，**将安全强化学习技术直接应用于LLM微调是不令人满意的。特别是，基于近端策略优化（PPO）的原始对偶方法训练成本高昂，并且存在强化学习的不稳定性**。

我们在这项工作的目标是开发一个更具可扩展性的微调框架，以提高LLM的安全性。我们没有采用PPO，而是**扩展了直接偏好优化（DPO）框架**，将其应用于受约束的微调。DPO作为一种稳定且轻量级的RLHF替代方案，通过直接从离线偏好数据集中优化策略，而不使用强化学习。它进一步消除了学习奖励函数或收集新的人类偏好数据的需要，尽管最近的研究表明，在应用DPO时，学习显式的奖励函数是有益的。在这项工作中，我们开发了一种基于DPO的对偶梯度下降方法，这种方法仍然需要预训练奖励和成本函数，但比中的原始对偶PPO方法更高效。使用Llama 2进行的评估表明，我们的受约束DPO（C-DPO）方法提供了DPO所缺失的强安全性保证，同时在相同的约束下获得了比受约束PPO方法更高的奖励。



## 预备知识

### 强化学习从人类反馈（RLHF）

RLHF 的正式框架通常涉及两个关键步骤：1）奖励建模；2）强化学习优化。

**奖励建模阶段**：奖励建模涉及学习一个奖励模型 $r$，以近似人类对偏好和不偏好答案的评估。在本工作中，我们采用最广泛使用的 Bradley-Terry（BT）模型：给定提示 $x$ 和回答 $y$，我们假设 $y$ 给定 $x$ 的逐点奖励为 $r(x, y)$，可以解释为生成偏好的真实奖励函数。那么，BT 模型将人类偏好分布 $p^*(y\_1 \succ y\_2\mid x)$ 表示为两个奖励之间的差值的函数：

$$
p^*(y_1 \succ y_2\mid x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} \quad (1)
$$

其中 $y\_1 \succ y\_2\mid x$ 表示在一对回答中，$y\_1$ 是偏好的，$y\_2$ 是不偏好的。假设我们有一个数据集 $D = \lbrace x\_i, y\_i^\omega \succ y\_i^l\rbrace \_{i=1}^N$，该数据集是从方程 (1) 中的相同分布中采样的，其中每个提示 $x$ 有一对答案 $(y^\omega, y^l)$，且 $y^\omega$ 比 $y^l$ 更受偏好。然后，我们可以参数化奖励模型 $r(x, y)$，并通过最小化以下逻辑回归损失来拟合参数：

$$
L(r; D) = -\mathbb{E}_{(x,y^\omega,y^l) \sim D}[\log(p(y^\omega \succ y^l\mid x))] \quad (2)
$$

$$
= -\mathbb{E}_{(x,y^\omega,y^l) \sim D}[\log \sigma(r(x, y^\omega) - r(x, y^l))] \quad (3)
$$

其中 $\sigma$ 是逻辑函数。

**强化学习优化阶段**：在强化学习阶段，我们通过以下目标函数来表述问题：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y\mid x)}[r(x, y)] - \beta D_{\text{KL}}[\pi_\theta(y\mid x) \mid \mid  \pi_{\text{ref}}(y\mid x)] \quad (4)
$$

该目标可以通过强化学习方法（如 PPO）或类似方法进行优化。此外，最近的研究也探索了无强化学习的范式，例如 DPO。

### 安全强化学习从人类反馈（Safe RLHF）

与上述传统的 RLHF 流程相比，Safe RLHF 范式增加了额外的步骤，以减少语言模型的潜在危害。

**成本建模阶段**：引入成本模型 $c$ 来区分安全和不安全的响应。给定数据集 $D = \lbrace x\_i, y\_i^\omega \succ y\_i^l, s\_i^\omega, s\_i^l\rbrace \_{i=1}^N$，其中 $y^\omega \succ y^l$ 表示 $y^l$ 比 $y^\omega$ 更安全，

$$
s(y) = \begin{cases} 
1 & \text{如果 } y \text{ 是不安全的} \\
-1 & \text{否则}
\end{cases}
$$

我们可以使用以下成对比较损失来学习成本模型，该损失与 BT 模型一致：

$$
L(c; D) = -\mathbb{E}_{(x,y^\omega,y^l) \sim D}[\log \sigma(c(x, y^\omega) - c(x, y^l))]
$$

$$
-\mathbb{E}_{(x,y^\omega,y^l,s^\omega,s^l) \sim D}[\log \sigma(s^\omega c(x, y^\omega)) + \log \sigma(s^l c(x, y^l))] \quad (5)
$$

**安全强化学习优化阶段**：在安全强化学习阶段，引入额外的约束条件，以确保优化策略 $\pi\_\theta(y\mid x)$ 的预期成本低于某个预定义的阈值 $C\_{\text{limit}}$：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y\mid x)}[r(x, y)] - \beta D_{\text{KL}}[\pi_\theta(y\mid x) \mid \mid  \pi_{\text{ref}}(y\mid x)] \quad (6)
$$

$$
\text{subject to } \mathbb{E}_{x \sim D, y \sim \pi_\theta(y\mid x)}[c(x, y)] \leq C_{\text{limit}} \quad (7)
$$

与传统的 RLHF 类似，此目标也可以通过 PPO 的变体（例如中采用的原始对偶 PPO 算法）进行优化。

## 方法

安全强化学习从人类反馈（Safe RLHF）框架提供了一种有前景的方法，用于优化语言模型（LMs），使其同时符合有用性和安全性的对齐目标。然而，基于强化学习（RL）的微调成本高昂且通常不稳定，并且在引入约束条件时情况会更加复杂。另一方面，原始的直接偏好优化（DPO）算法不能直接用于优化方程 (6) 中的 Safe RLHF 目标。因此，我们提出了受约束的直接偏好优化（Constrained DPO, C-DPO），它结合了对偶梯度下降和 DPO，以获得一种无需使用强化学习即可实现 Safe RLHF 目标的高效且轻量级的解决方案。

为了解决约束问题 (6)，首先通过拉格朗日方法将其转换为无约束形式。定义目标函数 $J\_r(\pi\_\theta) = \mathbb{E}\_{x \sim D, y \sim \pi\_\theta(y\mid x)}[r(x, y)] - \beta D\_{\text{KL}}[\pi\_\theta(y\mid x) \mid \mid  \pi\_{\text{ref}}(y\mid x)]$ 和约束函数 $J\_c(\pi\_\theta) = \mathbb{E}\_{x \sim D, y \sim \pi\_\theta(y\mid x)}[c(x, y)] - C\_{\text{limit}}$。然后，我们可以定义相关的拉格朗日函数 $J(\pi\_\theta, \lambda)$ 和对偶函数 $g(\lambda)$，如下所示：

$$
J(\pi_\theta, \lambda) = J_r(\pi_\theta) - \lambda J_c(\pi_\theta) \quad (8)
$$

$$
g(\lambda) = J(\pi^*_\lambda, \lambda) \quad (9)
$$

其中 $\pi^\ast\_\lambda = \arg\max\_{\pi\_\theta} J(\pi\_\theta, \lambda)$。这里，$\pi\_\theta$ 是原始变量，而 $\lambda \geq 0$ 是对偶变量。众所周知，对偶函数 $g(\lambda)$ 是原始问题 (6) 的上界。此外，可以证明目标函数在 $\pi\_\theta$ 上是凹的，因此强对偶性（strong duality）成立，即 $g(\lambda)$ 的最小值等于原始问题 (6) 的最大值。因此，可以通过对偶梯度方法来解决原始问题，该方法通过迭代执行以下两个步骤：1）基于当前的 $\lambda$ 值找到 $\pi^\ast\_\lambda = \arg\max\_{\pi\_\theta} J(\pi\_\theta, \lambda)$；2）对对偶函数 $g(\lambda)$ 进行梯度下降。

**给定 $\lambda$ 时找到最优的 $\pi^*_\lambda$**：算法从初始的 $\lambda$ 开始，解决无约束问题：

$$
\arg\max_{\pi_\theta} J(\pi_\theta, \lambda) \quad (10)
$$

可以证明无约束问题的最优解具有以下形式：

$$
\pi^*_\lambda(y\mid x) = \frac{1}{Z_\lambda(x)} \pi_{\text{ref}}(y\mid x) \exp\left(\frac{1}{\beta} (r(x, y) - \lambda c(x, y))\right) \quad (11)
$$

其中 $Z\_\lambda(x) = \sum\_y \pi\_{\text{ref}}(y\mid x) \exp\left[\frac{1}{\beta} (r(x, y) - \lambda c(x, y))\right]$ 是一个归一化函数。

我们现在定义一个新的奖励函数 $r\_\lambda(x, y) = r(x, y) - \lambda c(x, y)$，它通过 $\lambda$ 确定的权衡将 LLM 的回答的有用性和有害性结合起来。然后，(11) 式表明，给定 $\lambda$ 时的最优策略可以直接从 $r\_\lambda$ 推导出来。然而，由于归一化函数难以计算，这在实际中很难实现。在 DPO 中，当 $\lambda = 0$ 时，通过将真实奖励 $r^\ast$ 用最优策略 $\pi^\ast$ 表示（根据 (11) 式），然后将其代入 BT 偏好模型 (1) 中，其中归一化函数会相互抵消。这样，最优策略可以通过最小化 (3) 式中的回归损失来获得。为了将这种方法适应到我们的设置中，关键思想是根据 $r\_\lambda(x, y)$ 定义一个新的 BT 偏好模型，如下所示：

$$
p^*_\lambda(y_1 \succ y_2\mid x) = \frac{\exp(r_\lambda(x, y_1))}{\exp(r_\lambda(x, y_1)) + \exp(r_\lambda(x, y_2))} \quad (12)
$$

其中 $y\_1 \succ y\_2\mid x$ 表示 $r\_\lambda(x, y\_1) = r(x, y\_1) - \lambda c(x, y\_1) \succ r\_\lambda(x, y\_2) = r(x, y\_2) - \lambda c(x, y\_2)$。然后，我们根据新的偏好模型生成一个新的数据集，其提示与原始数据集 $D$ 中的相同，而回答则是根据新的偏好模型采样的。这一步是必要的，因为原始数据集 $D$ 中的提示回答对的偏好通常不是根据我们的偏好函数 $r\_\lambda$ 生成的。幸运的是，如果我们有预训练的奖励和成本函数，我们可以计算原始数据集中每个提示回答对的奖励和成本，然后根据 (12) 式生成一个新的偏好数据集 $D\_{r\_\lambda}$。

在得到新的偏好数据集 $D\_{r\_\lambda}$ 后，我们可以像 DPO 那样，为给定的 $\lambda$ 优化策略 $\pi\_\theta$，并制定以下类似于 DPO 的最大似然目标：

$$
L(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y^\omega,y^l) \sim D_{r_\lambda}} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y^\omega\mid x)}{\pi_{\text{ref}}(y^\omega\mid x)} - \beta \log \frac{\pi_\theta(y^l\mid x)}{\pi_{\text{ref}}(y^l\mid x)}\right) \right] \quad (13)
$$

请注意，这个目标与原始的优化问题 (10) 并不相同。然而，我们可以证明，如果新的 BT 偏好模型 $p^\ast\_\lambda$ 是根据我们的偏好函数 $r\_\lambda = r(x, y) - \lambda c(x, y)$ 生成的，并且新的数据集 $D\_{r\_\lambda}$ 足够大且完美地拟合新的偏好 $p^\ast\_\lambda$，使得可以通过最小化 $L(r; D\_{r\_\lambda})$ 得到 $r\_\lambda$，那么最小化 (13) 式的最优策略与最大化原始目标 (10) 的最优策略是一致的。

**对 $\lambda$ 进行梯度下降**：接下来，我们对对偶函数 $g(\lambda)$ 应用梯度下降，以更新 $\lambda$ 并最小化对偶函数 $g(\lambda)$。如附录 A.2.4 所示，对偶函数 $g(\lambda)$ 的梯度可以表示为：

$$
\frac{dg(\lambda)}{d\lambda} = \mathbb{E}_{x \sim D, y \sim \pi^*_\lambda(y\mid x)}[C_{\text{limit}} - c(x, y)] \quad (14)
$$

它表示所学策略 $\pi^\ast_\lambda$ 的预期约束违反情况。

## 实验

**基准模型**：除了 C-DPO，我们还考虑了基于预训练的 Llama-27B 模型的几种基准模型，包括使用监督微调（SFT）微调的模型、原始的直接偏好优化（DPO），以及 Beaver-v1，后者是使用原始对偶 PPO 训练的 Safe-RLHF 方法。

**评估方法**：从 BEAVERTAILS 测试数据集中随机抽取 2000 个提示，为每个提示生成 5 个回答，并计算这些回答的预期奖励和成本，以及它们的方差。

**实验结果**：表 1 展示了不同模型在测试数据集上的性能比较，重点关注有用性和无害性。所有评估模型的一个显著观察结果是性能指标的方差很大，尤其是在成本领域，这表明模型生成的回答存在不稳定性。原因之一是允许的奖励和成本值范围较大。回答的多样性导致成本和奖励值存在较大差距。此外，当前的优化目标侧重于策略的预期成本，而对整体方差的关注较少。尽管如此，我们发现 **SFT 模型倾向于生成有害性较高且奖励相对较低的回答，而 DPO 获得了最高的奖励，但也承受了最高的成本，因为它旨在提高有用性，这通常与无害性相冲突**。Beaver-v1 显著提高了无害性，但导致奖励大幅下降。相比之下，C-DPO（$\lambda = 1$）在奖励方面优于 Beaver-v1，同时在成本方面仅略有增加，实现了在不违反安全约束的情况下最大化预期奖励的主要目标。此外，C-DPO（$\lambda = 0.4$）在当前 $C_{\text{limit}} = 0$ 的情况下成为最优策略，超过了 C-DPO（$\lambda = 1$）的奖励，同时遵守安全约束。这表明我们在算法中对对偶变量 $\lambda$ 的微调有助于在有用性和有害性之间实现有效的权衡。

![image-20250409142030283]({{site.url}}/img/2025-4-09-Enhancing_LLM_Safety_via_Constrained_Direct_Preference_Optimization/image-20250409142030283.png)

## 附录

**A.1 讨论**  

**A.2 分析结果**  

- A.2.1 安全强化学习的强对偶性  
- A.2.2 无约束目标的最优解推导  
- A.2.3 安全强化学习与最大似然目标的等价性  
- A.2.4 对偶函数的梯度推导  

**A.3 相关工作**  

**A.4 关于受约束直接偏好优化（C-DPO）算法的详细信息**  

**A.5 关于实验设置的详细信息**  

- A.5.1 数据集  
- A.5.2 预训练的奖励和成本函数  
- A.5.3 实现细节  
- A.5.4 评估细节  

**A.6 额外的实验结果**  

- A.6.1 训练曲线  
- A.6.2 额外的测试性能  
- A.6.3 不同模型生成的样本  

