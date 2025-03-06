---
layout:     post
title:      Bridging Large Language Models and Optimization - A Unified Framework for Text-attributed Combinatorial Optimization
subtitle:   LLM结合AM
date:       2025/03/06
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - LLM
---

Bridging Large Language Models and Optimization - A Unified Framework for Text-attributed Combinatorial Optimization

荷兰埃因霍温理工大学

2024.12.15 arxiv


以下是省流版

## 方法

![image-20250306164429323]({{site.url}}/img/2025-3-06-Bridging_Large_Language_Models_and_Optimization_A_Unified_Framework_for_Text-attributed_Combinatorial_Optimization/image-20250306164429323.png)

整体结构图如上，说实话不是很看得懂。

作者针对TSP和KP问题进行研究：

- 将（问题描述，节点信息）输入到 LLM 当中，后者会加上诸如 TSP 的 3 个临近点的信息，由 LLM 获得问题的embedding $k^P$ 和节点的 embedding $\lbrace v^P_i\rbrace$。

- 这些embedding的维度很高，需要经过一个线性层降低维度。

- 节点的embedding直接输入AM的编码器，取代原本AM中二维坐标到embedding的映射层。

- 在解码器中，上下文信息从原本的 $[c\_t^P,h\_1^{(N)},h\_t^{(N)}]$，$c\_t^P$ 是如CVRP中的车辆容量，KP的背包容量，$h\_1^{(N)},h\_t^{(N)}$ 是当前部分解的第一个和最后一个节点的嵌入。现在额外加上 $k^P$。

- 训练的时候冻结大模型的参数。

- 由于是TSP和KP同时训练，假设两个问题的梯度分别是 $g\_i$ 和 $g\_j$，需要确认两个梯度是否冲突，$\delta=g\_i\times g\_j$。

  $$
  \hat{\mathbf{g}}_i= \begin{cases}\mathbf{g}_i-\frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\left\|\mathbf{g}_j\right\|^2} \mathbf{g}_j, & \text { if } \delta<0 \\ \mathbf{g}_i, & \text { if } \delta \geq 0\end{cases}
  $$

  然后用 $\hat{\mathbf{g}}_i$ 作为梯度更新。这个方法叫做CGERL（Conflict Gradient Erasing Reinforcement Learning，冲突梯度消除强化学习）

## 实验

在 TSP、CVRP、KP、MVCP（最小顶点覆盖）、SMTWTP（单机总加权延迟问题）训练。

新任务微调：VRPB（带回程的VRP）、MISP（最大独立集）

baseline：

- 基于LLM的优化方法：AEL（LLM进化算法）、ReEvo（反思进化）、SGE（自我引导搜索）、ACO（ReEvo应用）、LMEA（进化算法）、OPRP（提示）

  ![image-20250306172119321]({{site.url}}/img/2025-3-06-Bridging_Large_Language_Models_and_Optimization_A_Unified_Framework_for_Text-attributed_Combinatorial_Optimization/image-20250306172119321.png)

- 传统求解器：OR-Tools、Gurobi

- 传统算法：TSP的最近邻法和最远插入法、CVPR的扫描启发式和并行saving、KP的贪心、MVCP的MVCApprox、SMTWTP的最早交期优先（EDD）规则调度，蚁群算法

  ![image-20250306172235854]({{site.url}}/img/2025-3-06-Bridging_Large_Language_Models_and_Optimization_A_Unified_Framework_for_Text-attributed_Combinatorial_Optimization/image-20250306172235854.png)

- 神经组合优化求解器：AM、POMO

  在附录，看不到

消融实验：

- 不同LLM：包括ST（句子变换器）、E5-large-v2（E5）、Llama2-7b、Llama-8b。越小越好。

  ![image-20250306172446901]({{site.url}}/img/2025-3-06-Bridging_Large_Language_Models_and_Optimization_A_Unified_Framework_for_Text-attributed_Combinatorial_Optimization/image-20250306172446901.png)

- CGERL的效果：对比的CGERL和REINFORCE，上图b

- TAI（编码器中的启发式信息）：上图b

- 任务之间的协同学习：附录，看不到

泛化实验：

- 在新COP上微调
- 在新的规模上微调，都在附录看不到

