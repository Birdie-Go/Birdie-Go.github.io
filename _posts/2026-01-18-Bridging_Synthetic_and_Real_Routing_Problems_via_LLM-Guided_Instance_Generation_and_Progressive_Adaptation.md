---
layout:     post
title:      Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation
subtitle:   LLM合成数据解决train和test数据分布不一致
date:       2026/01/18
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - AAAI
---

Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation

zhiguang cao课题组

AAAI2026



## 研究背景和问题

近年来，基于深度强化学习的神经求解器在**合成数据**（如均匀分布的TSP或CVRP）上表现优异，但在**真实世界基准数据集**（如TSPLib、CVRPLib）上泛化能力较差。原因在于：

- 合成数据分布与真实数据分布差异大（distributional shift）
- 真实数据具有更复杂的空间结构（如聚类、重复模式、非均匀分布）

## 贡献

提出框架：EvoReal 框架。

EvoReal 通过**大模型引导的演化式数据生成 + 渐进式微调**，桥接合成数据与真实数据之间的鸿沟。

1. **LLM-guided Generator Evolution（大模型引导的数据生成器演化）**

- 使用大语言模型（如 OpenAI o3）**演化式地设计数据生成器**
- 每个生成器是一个 Python 函数，输出具有特定结构特征的 TSP/CVRP 实例
- 通过**代理评估（proxy evaluation）**快速评估生成器质量（如在验证集上的平均gap）
- 采用**交叉、变异、反射等演化机制**不断优化生成器，使其输出分布逼近真实数据

2. **Progressive Fine-tuning（渐进式微调）**

- **Phase 1**：用演化出的生成器生成大量合成数据，对预训练模型进行微调，使其适应多样化结构
- **Phase 2**：再用真实基准数据（如TSPLib）进一步微调，完成从“合成”到“真实”的平滑过渡

## 方法

提出一个新颖的 **由大语言模型引导的演化框架 EvoReal**，用于演化 VRP 数据生成器，使其生成与 VRPLib 风格一致的分布；并配套提出 **渐进式微调策略**，逐步将预训练神经模型从合成均匀实例迁移到真实 VRPLib 实例。

所提出的 LLM 引导演化方法主要由两个组件构成：

1. **LLM 驱动的演化搜索**：逐步发展出具备结构特异性的生成器；
2. **代理评估（proxy evaluation）**：通过训练过程中的验证结果评估每个生成器的性能。

以下以 TSP 为例，详细阐述生成器演化与渐进式微调框架。

### 生成器模块化

下图展示了整个流程。对于 TSP，演化过程由三种类型的生成器驱动，每种生成器对应一类 TSP 分布，确保生成的分布能够模拟 TSPLib 的风格。我们将这三种演化生成器合并，用于生成混合分布的 TSP 数据以进行微调。

![image-20260118202157690]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202157690.png)

为了保证问题规模和分布的多样性，我们从 TSPLib 中选取 70 个规模小于 5000 的问题，其中 48 个（约 70%）用于验证，剩余 22 个作为未见测试集。进一步将 48 个验证问题划分为三类（S1、S2、S3），每类作为特定生成器的验证集。CVRP 类似地从 CVRPLib 的 SetX 中随机选取 70% 用于验证与直接微调，30% 作为测试集。

### 生成器演化

我们提示 LLM 将潜在生成器配置转换为真实路径分布。形式化地，令生成器为：

$$
h_\phi: \mathcal{Z} \to \mathcal{X}
$$

其中 $\phi$ 为 LLM 参数，$z \sim \mathcal{Z}$ 为潜在变量（如随机种子），$x = h_\phi(z)$ 为生成的合成实例。演化目标为最小化生成分布与真实分布之间的差异：

$$
\min_\phi \mathcal{L}_{\text{evolve}}(\phi) = D\left(\mathcal{D}_{\text{real}} \,\mid \, \mathcal{D}(h_\phi(\mathcal{Z}))\right)
$$

其中 $D(\cdot \mid \cdot)$ 为散度度量（如 KL 散度或平均 gap）。

#### 生成器表示

每个生成器 $h$ 由三部分组成：

1. 函数描述：定义输入格式与输出要求；
2. 实现代码：PyTorch 或 NumPy 实现；
3. 适应度分数：在验证集上的平均 gap，作为性能指标。

#### 代理评估

我们采用外部代理指标（如 gap）作为散度度量 $D$，代替人工或参考评估。每个生成器仅微调模型少量 epoch（如 10 个），并记录验证集上的最佳平均 gap 作为适应度分数。分数越低，生成器越优。

#### 种群初始化

初始种群由 LLM 根据提示生成，提示包括：

- 问题描述
- 函数签名
- 系统约束
- 种子生成器示例
- 设计引导（外部知识）

#### 演化流程

我们对 ReEvo 框架进行改进，引入：

- **基于排名的选择**：适应度越低，存活概率越高；
- **延迟选择**：在交叉与变异后进行选择，避免过早淘汰多样性。

演化步骤如下：

![image-20260118202547412]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202547412.png)

1. 选择生成器类型，初始化种群；
2. 通过交叉、变异、反射扩展种群；
3. 代理评估新生成器，记录适应度；
4. 按排名选择个体，维持种群规模；
5. 重复步骤 2–4；
6. 达到最大评估次数或连续无改进时停止。

### 渐进式微调

演化完成后，我们采用两阶段渐进微调策略：

#### 阶段一

使用所有类型最优生成器**联合生成大规模合成数据**，对预训练模型进行微调。此阶段模型接触多样化、TSPLib 风格的结构模式，为后续真实数据微调做准备。

#### 阶段二

在阶段一基础上，**直接使用 TSPLib 实例进行微调**，实现从“合成”到“真实”的平滑迁移。两阶段结合，**同时弥合分布差距与规模差距**。

### 总结

| 步骤       | 目标                 | 方法                    |
| ---------- | -------------------- | ----------------------- |
| 生成器演化 | 自动发现真实分布结构 | LLM 引导演化 + 代理评估 |
| 阶段一微调 | 适应多样化结构       | 用演化生成器数据微调    |
| 阶段二微调 | 迁移到真实数据       | 用 TSPLib/CVRPLib 微调  |

## 实验

演化阶段使用 **OpenAI o3** 模型，单卡 RTX 3090 Ti。微调阶段使用 **RTX A5000 40 GB**。

非神经求解器设置最大运行时间：

- LKH-3 与 OR-Tools：*T*max=1.8*n*  秒
- Concorde：*T*max=10  分钟

TSPLib

![image-20260118202635236]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202635236.png)

CVRPLib

![image-20260118202816885]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202816885.png)

消融

![image-20260118202835627]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202835627.png)

![image-20260118203006004]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118203006004.png)

本文的演化生成器和五种朴素分布（Beta、指数、二项、高斯混合、高斯条带）

![image-20260118202918631]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118202918631.png)

22个未见分布的评估

![image-20260118203114017]({{site.url}}/img/2026-01-18-Bridging_Synthetic_and_Real_Routing_Problems_via_LLM-Guided_Instance_Generation_and_Progressive_Adaptation/image-20260118203114017.png)



