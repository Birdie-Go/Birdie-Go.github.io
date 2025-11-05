---
layout:     post
title:      Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning
subtitle:   NIPS2025 EoH驱动的映射算子做大规模
date:       2025/11/05
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - NIPS
---

Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning

王振坤课题组

NIPS2025

代码：https://github.com/CIAM-Group/TTPL

## 摘要

神经组合优化（NCO）已成为一种基于学习的有前景的范式，通过最小化对手工设计的依赖来解决车辆路径问题（VRP）。尽管现有的 NCO 方法在小规模实例（例如 100 个节点）上训练并在类似规模的问题上表现出相当大的成功，但当应用于大规模场景时，它们的性能显著下降。这种下降源于训练和测试数据之间的分布偏移，使得在小实例上学习到的策略对更大问题无效。为了克服这一限制，我们引入了一种由大语言模型（LLM）驱动的新学习框架。该框架学习训练和测试分布之间的投影，然后部署该投影以增强 NCO 模型的可扩展性。值得注意的是，与当前需要在测试阶段与神经网络联合训练的技术不同，我们的方法仅在推理阶段运行，无需重新训练模型。大量实验表明，我们的方法使骨干模型（在 100 节点实例上训练）在来自不同分布的最多 100K 节点的大规模旅行商问题（TSP）和容量约束车辆路径问题（CVRP）上实现卓越性能。



## 引言

难点：

由于小规模和大规模实例之间的分布差异，现有方法在处理大规模问题（例如，超过 1K 个节点的问题）时效果显著降低，从而严重限制了它们的实际应用能力。为了解决可扩展性问题，一些尝试致力于在更大规模的 VRP 实例（即，具有 500 个节点的实例）上训练 NCO 模型。然而，现有的监督学习（SL）和强化学习（RL）在大规模 VRP 实例训练时均显示出其不足之处。SL 缺乏足够的标签（例如，高质量解），而 RL 则受到极其稀疏的奖励信号影响。

现有方法：

- 当前方法专注于在小规模实例上训练模型，并通过分解或局部策略将其推广到大规模场景。基于分解的方法首先将大规模问题简化为一组子问题。随后，可以利用在小规模实例上训练的求解器来构建这些子问题的部分解。
  - 缺点：然而，分解的范围通常需要手动调整，并且分解问题可能会改变其性质，从而损害求解算法的最优性。
- 另一种努力是局部策略方法。它首先在每个构造步骤将搜索空间缩小到基于欧氏距离到最后访问节点的小候选子图。下一个节点由原始策略或局部策略决定。然而，所选子图的分布通常与训练实例不同，这显著损害了模型的可扩展性。当前方法采用投影技术，将这些不同的输入分布有效地转换为训练期间遇到的均匀分布。
  - 缺点：然而，现有策略需要与模型训练集成，以确保在测试期间的有效性。同时，这些手动设计的策略严重依赖专门的领域专业知识，从而限制了它们的适应性。

贡献：

为了应对这一挑战，我们提出了一种新的学习框架，称为测试时投影学习（TTPL），由大语言模型（LLM）驱动，以设计高效的投影策略。特别是，我们利用 LLM 来学习输入子图与训练实例之间的关联，从而开发出增强模型泛化的投影策略。与现有工作不同，我们的方法可以直接应用于推理阶段，无需从头开始训练模型。此外，我们提出了多视图决策融合（MVDF）模块来提高模型泛化能力。具体来说，我们对子图进行数据增强以生成多个视图。这些视图随后由模型处理，每个视图产生不同的节点选择概率。最后，这些概率被聚合，选择具有最高结果置信度得分的节点。

在合成和真实世界基准测试上对 TSP 和 CVRP 进行了全面实验。结果表明，我们提出的分布适配器使基础模型能够在无需额外微调的情况下，在大多数大规模 VRP 测试实例上实现卓越性能，该基础模型预训练于小规模实例（例如，100 个节点）。我们的消融研究验证了所设计投影的有效性。

## 相关工作

- 直接泛化（小规模训练然后泛化到大规模，或者直接在中规模上训练）
  - BQ（修改了MDP）在100上训练，可以泛化到1000
  - LEHD（轻编码器重解码器）在100上训练，可以泛化到1000
  - Towards omni-generalizable neural methods for vehicle routing problems、Pointerformer、ICAM、Dan在500上训练
  - Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems在1000上训练
  - 缺点：搜索空间呈指数增长，带来了极高的计算成本
- 基于分解的泛化（分解成子问题，求解，然后合并）
  -  Learning to delegate for large-scale vehicle routing（nips21）
  - Rbg: Hierarchically solving large-scale routing problems in logistic systems via reinforcement learning.（kdd22）
  - Generalize learned heuristics to solve large-scale vehicle routing problems in real-time.（nips24）
  - UDC: A unified neural divide-and-conquer framework for large-scale combinatorial optimization problems.（nips24）
  - H-tsp: Hierarchically solving the large-scale traveling salesman problem.（aaai23）
  - Glop: Learning global partition and local construction for solving large-scale routing problems in real-time.（aaai24）
  - 缺点1：求解其他复杂的 VRP（例如，CVRP）时，这种分解变得难以处理，无法通过单个策略或模型实现
  - 缺点2：当分解的子问题不够小时，可能仍需要其他泛化技术来求解
- 基于局部策略的泛化（每一步将搜索空间缩减为从最后访问节点出发的 K 最近邻）
  - 结合辅助距离信息： Distance-aware attention reshaping: Enhance generalization of neural solver for large-scale vehicle routing problems.
  - 局部策略：Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy.
  - invit、l2r、bq：直接在邻域里面选
  - glop、invit、Generalize a small pre-trained model to arbitrarily large tsp instances.：投影

## 预备知识

- TSP和CVRP介绍

- AHD（自动启发式发现）：给定目标任务 $T$ 和输入空间 $X_T$，自动发现高性能启发式 $h$

  $$
  h^*=\arg\max_{h\in H}\mathbb{E}_{x\sim X_T}f_T(x,h),
  $$

- 基于LLM的AHD：EoH



## 大规模VRP的分布投影

![image-20251105142658190]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105142658190.png)

抽取出的 KNN 子图具有多种分布，与训练实例的分布差异巨大，导致模型难以预测下一个有潜力的节点。

![image-20251105145447520]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105145447520.png)

用LEHD，应用了invit的投影方式（单纯拉伸到1x1平面）。

结论：开发更先进、更鲁棒的投影方法对提升模型泛化能力至关重要。



## 方法

![image-20251105161258562]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105161258562.png)

### 测试时投影（Test-Time Projection Learning，TTPL）

包含两个组件：

1. 基于 NCO 的策略评估器：衡量 LLM 生成策略的效果；
2. 进化投影策略生成器：采用多样化提示策略引导 LLM 开发新策略。

![image-20251105150029848]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105150029848.png)

输入包括：种群大小 $N$，进化代数 $N_g$，NCO 模型 $f_\theta$，用于进化策略的 LLM，评估数据集 D$。

TTPL 遵循 EoH 的一般框架：

- 首先，使用 LLM 生成 $N$ 个个体（每个个体包含自然语言描述与实现投影的代码块），并用 $f\_\theta$ 在 $D$ 上评估，构建初始种群 $P\_0$。
- 随后进行 $N_g$ 代迭代：每代根据适应度排序，采用四种提示策略（E1, E2, M1, M2）生成后代，评估后保留前 $N$ 个最优个体。
- 最终输出历代找到的最优投影策略 $a^*$。

个体评估方案：将其嵌入目标 NCO 模型，并在评估数据集 $D$ 上计算目标值。具体步骤：

1. 对当前节点 $v\_i$，选取其 $k$ 个最近邻构成子图 $G\_k$；
2. 应用策略对 $G\_k$ 进行投影，得到 $G\_N$；
3. 将 $G\_N$ 输入 NCO 模型 $f\_\theta$，选出下一跳节点；
4. 重复上述过程，逐步构造出完整解，记录解长度作为该策略的适应度。

个体生成：初始化后，进行 $N_g$ 代进化。每代

1. 按排名概率采样父代：

   $$
   \mathrm{prob}_i=\frac{1/2^{r_i}}{\sum_{j=1}^N 1/2^{r_j}},
   $$

其中 $r_i$ 为个体 $i$ 的适应度排名；
2. 使用四种提示策略（E1, E2 探索；M1, M2 修改）生成后代；
3. 评估后代并保留前 $N$ 个最优个体形成下一代。

### 多视图决策融合（Multi-View Decision Fusion，MVDF）

MVDF 通过对子图进行几何变换生成 $M$ 个增广视图，分别输入 NCO 模型得到对应 logit $\mathbf{l}_m\in\mathbb{R}^K$，再聚合为最终选择概率：

$$
\mathbf{p}=\sigma\!\left(\sum_{m=1}^M \mathbf{l}_m\right),
$$

其中 $\sigma(\cdot)$ 为 softmax。该集成策略迫使模型学习变换不变特征，有效抵消局部密度偏差。

## 实验

问题设置：

![image-20251105154204130]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105154204130.png)

Baseline：

- 经典求解器：Concorde、LKH3、HGS
- 插入启发式：随机插入
- 构造NCO：POMO、BQ、LEHD、INVIT、SIGD
- 热图：DIFUSCO
- 分解：GLOP、H-TSP
- 局部构造：ELG

RRC是随机重构

![image-20251105154421120]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105154421120.png)

![image-20251105154429614]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105154429614.png)

### 消融实验

- 投影的设计在上述已经证明是有效的

- 需要验证多视图决策融合的作用

![image-20251105164630127]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105164630127.png)

感觉放错图了，这俩图一模一样。

### 功能性实验

用pomo

![image-20251105164902378]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105164902378.png)

多分布

![image-20251105164844573]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105164844573.png)

## 限制和未来工作

LLM用于寻找合适的映射算法的时间很久，未来工作希望客服这一困难。

## 附录

A 相关工作的补充

B 方法的细节

- EoH的prompt
- EoH算法

C LLM写的算法及其数学推导

![image-20251105165243652]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165243652.png)

每个数据集有一个映射算法。

D 实验

- TSPLIB、CVRPLIB的实验

E 实验分析

- 在pomo、lehd、bq、sigd上都用了这个方法

![image-20251105165439155]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165439155.png)

- llm学习写映射算法的时间

![image-20251105165506527]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165506527.png)

- 不同llm的实验

![image-20251105165522625]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165522625.png)

F 可视化

![image-20251105165553527]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165553527.png)

![image-20251105165559604]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165559604.png)

![image-20251105165605756]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165605756.png)

![image-20251105165611377]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165611377.png)

![image-20251105165623937]({{site.url}}/img/2025-11-05-Improving_Generalization_of_Neural_Combinatorial_Optimization_for_Vehicle_Routing_Problems_via_Test-Time_Projection_Learning/image-20251105165623937.png)

