---
layout:     post
title:      Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM
subtitle:   ICLR2026under review LLM优化HGS
date:       2025/11/22
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - ICLR
---

Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM

ICLR2026 under review

zhiguang cao课题组



## 摘要

尽管大语言模型（LLMs）正逐渐成为解决车辆路径问题（VRP）的自动化启发式设计工具，但现有最先进的方法主要依赖于像 GPT-4 这样的大型通用模型。本文挑战了这一范式，提出通过微调小型的、专门化的 LLM，也能生成超越专家设计的启发式组件，并在先进求解器中表现出色。我们提出了一种新颖的强化学习（RL）框架 **RFTHGS**，用于微调一个小型 LLM，使其能够生成高性能的交叉算子，嵌入到混合遗传搜索（HGS）求解器中，用于解决容量限制车辆路径问题（CVRP）。我们的方法采用了一个多层次的、基于课程学习的奖励函数，逐步引导 LLM 先生成可编译的代码，然后生成可执行的算子，最终生成超越人类专家设计的组件。此外，我们还引入了一个算子缓存机制，与奖励函数协同工作，防止抄袭行为并在训练过程中鼓励多样性。实验结果表明，经过微调的 LLM 生成的交叉算子在 HGS 中显著优于人类专家设计的算子。这一性能优势在小规模问题上得到了验证，并可泛化到多达 1000 个节点的大规模问题。此外，RFTHGS 超越了领先的神经组合优化基线方法、基于提示的方法以及商用 LLM（包括 GPT-4o 等）。

## 引言

- 早期研究用LLM作为端到端求解器直接生成 VRP 的解。然而，这类纯生成式方法往往生成的解质量远低于传统或深度学习方法，且常常不可行。
- 另一种思路是逆转这一关系，不再让进化计算为 LLM 提供框架，而是利用进化机制引导 LLM 生成更优的启发式算法，如EoH和ReEvo，通过进化选择机制迭代优化 LLM 生成的启发式规则。
- 还有一类研究则更注重通用性，尝试构建适用于多种 VRP 变体的通用框架，虽然牺牲了部分性能以换取更强的泛化能力。例如，ARS利用预定义结构生成约束检查函数，DRoC则通过检索机制调用外部求解器（如 OR-Tools）生成代码。

挑战：我们是否能够通过微调小型 LLM，优化这些求解器中的关键组件，从而实现超越专家水平的性能？

## 相关工作

- LLM的推理能力发展：思维链 -> 自一致性和思维树 -> 强化学习。混合方法包括结合MCTS、RAG等。
- 基于LLM的CVRP求解
  - ARS：利用 LLMs 自动生成带约束感知的启发式规则，通过将自然语言描述合成为可执行代码
  - Hercules：采用核心抽象提示，从精英解中抽象出核心组件以推导高性能启发式算法
  - EoH：通过进化选择机制迭代优化 LLM 生成的启发式规则
  - CALM：进一步将 LLM 的强化学习微调嵌入进化循环中，使模型与其生成的启发式规则共同演化

## 预备知识

### HGS

HGS 是经典遗传算法的扩展，其特点是将**基于种群的进化搜索**与**强化的局部改进过程**紧密结合。其核心机制是维护一个多样化的解种群，并通过选择、交叉和局部搜索等操作不断迭代优化。在该框架中，**交叉算子**是实现全局探索的主要机制。

### GRPO

群组相对策略优化（Group Relative Policy Optimization, GRPO）是近端策略优化（PPO）算法的一种改进版本）。

GRPO 的关键创新在于使用**归一化的奖励函数**来计算优势值，其中均值和方差通过对当前策略 $\pi_k(\cdot \mid x)$ 在每个输入提示 $x$ 下采样一组大小为 $G$ 的样本进行蒙特卡洛估计。

对于给定的参数 $\varepsilon, \beta > 0$ 和参考策略 $\pi_{\text{ref}}$（通常是基础模型），GRPO 的目标优化问题定义如下：

$$
\max_{\pi} \mathbb{E}_{y \sim \pi_k(\cdot \mid x)} \min\left[
\frac{\pi(y \mid x)}{\pi_k(y \mid x)} A_{\pi_k}(x,y),\;
\text{clip}\left(\frac{\pi(y \mid x)}{\pi_k(y \mid x)}, 1 - \varepsilon, 1 + \varepsilon\right) A_{\pi_k}(x,y)
\right] - \beta \, \text{KL}(\pi \ \mid  \pi_{\text{ref}})
$$

其中：

- $\text{KL}$ 表示 Kullback-Leibler 散度；
- $A_{\pi_k}(x, y_i)$ 表示 GRPO 的优势函数，定义为：

$$
A_{\pi_k}(x, y_i) = \frac{r(x, y_i) - \mathbb{E}_{\pi_k}[r(x, y_i)]}{\sqrt{\mathbb{E}_{\pi_k}\left[(r(x, y_i) - \mathbb{E}_{\pi_k}[r(x, y_i)])^2\right] + \varepsilon}}
\simeq \frac{r_i - \mu(\{r_\ell\})}{\sqrt{\sigma^2(\{r_\ell\}) + \varepsilon}},\quad 1 \leq \ell \leq G
$$

其中：

- $r(x, y_i)$ 是第 $i$ 个响应对应的奖励；
- $\mu$ 和 $\sigma$ 分别表示从当前策略中采样得到的奖励的经验均值和标准差；
- $G$ 是每个输入提示 $x$ 下采样的响应数量（即组大小）。

## 方法

### 单步POMDP建模

我们将算子优化问题建模为一个单步部分可观察马尔可夫决策过程（One-Step POMDP），定义如下：

- **状态（State）**：状态 $X \in \mathcal{X}$ 表示 LLM 接收到的提示（tokenized prompt），包含任务指令和目标算子的示例。为避免上下文过载，输入仅限于目标算子本身，而非整个求解器库，因此模型只能观察到部分状态空间。

- **动作（Action）**：动作为 LLM 生成的响应 $Y \sim p(\cdot \mid X)$，包括两个部分：  
  1. 推理部分，用特殊 token `<think>` 和 `</think>` 包围；  
  2. 优化后的算子代码。

- **状态转移（State Transition）**：动作生成后状态终止，不允许进一步优化，因此这是一个单步 POMDP。

- **奖励（Reward）**：奖励 $r \in \mathbb{R}$ 是一个标量，用于评估 LLM 生成的算子的质量。

- **策略网络（Policy Network）**：策略网络是一个基础大语言模型，记为 $\pi_\theta(\cdot \mid X)$，参数为 $\theta$，可采样生成优化算子。本文使用的是参数量为14B的小型 LLM。

### 多维度奖励设计与防抄袭缓存

设计了一个多层次的奖励函数，遵循课程学习原则，逐步引导 LLM 生成超越人类专家的算子。奖励函数分为三个阶段：

#### 奖励函数定义：

$$
r(o) = 
\begin{cases}
-1 & \text{若 } o \notin \mathbb{C} \\
-0.8 & \text{若 } o \in \mathbb{C}, o \notin \mathbb{E} \\
-0.9 & \text{若 } o \in \mathbb{C}, o \in \mathbb{E}, o \in \mathbb{P} \\
\max\left(-0.7, \frac{\phi_{\text{HGS}}^J(o_{\text{expert}}) - \phi_{\text{HGS}}^J(o)}{\phi_{\text{HGS}}^J(o_{\text{expert}})}\right) & \text{若 } o \in \mathbb{C}, o \in \mathbb{E}, o \notin \mathbb{P}
\end{cases}
$$

其中：

- $o$：生成的算子  
- $\mathbb{C}$：可编译代码集合  
- $\mathbb{E}$：可执行代码集合  
- $\mathbb{P}$：抄袭代码集合  
- $\phi_{\text{HGS}}^J(o)$：在 $J$ 个 CVRP 实例上运行算子 $o$ 的平均目标值  

#### 防抄袭机制：

- 使用抽象语法树（AST）对生成的算子进行结构匹配，若与缓存中的示例过于相似，则给予惩罚，鼓励多样性。

#### 增量编译加速：

- 每次只重新编译被修改的代码及其依赖项，将编译时间缩短至全量编译的约25%。

### 强化学习算法：DAPO

使用 **DAPO**（一种改进的 GRPO 算法）进行策略优化，主要改进包括：

1. **Clip-Higher 机制**：
   使用不同的上下界裁剪比率，鼓励模型对低概率的优质 token 给予更高的概率，增强探索性。

2. **Token 级策略梯度损失**：
   对每个 token 计算梯度并平均，而非对整个响应平均，提升长推理链的训练效果。

3. **过长响应处理**： 
   - **Overlong Filtering**：截断的响应不参与训练；  
   - **Soft Overlong Punishment**：对过长但仍有效的响应给予渐进式惩罚，鼓励简洁表达。

## 实验

配置

- Qwen-14B
- batch：16，group：16

![image-20251122172356382]({{site.url}}/img/2025-11-22-Refining_Hybrid_Genetic_Search_for_CVRP_via_Reinforcement_Learning-Finetuned_LLM/image-20251122172356382.png)

![image-20251122172432897]({{site.url}}/img/2025-11-22-Refining_Hybrid_Genetic_Search_for_CVRP_via_Reinforcement_Learning-Finetuned_LLM/image-20251122172432897.png)
