---
layout:     post
title:      ReinforceNS Reinforcement Learning-based Multi-start Neighborhood Search for Solving the Traveling Thief Problem
subtitle:   IJCAI2024 TTP问题
date:       2024/10/24
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - IJCAI
---

ReinforceNS: Reinforcement Learning-based Multi-start Neighborhood Search for Solving the Traveling Thief Problem

基于强化学习的多起点邻域搜索求解旅行小偷问题

杭州电子科技大学和南京电子工程研究所



## 摘要

旅行贼问题（TTP）是一个具有挑战性的组合优化问题，具有广泛的实际应用。TTP结合了两个np困难问题：旅行商问题（TSP）和背包问题（KP）。虽然针对TSP和KP已经开发了许多基于机器学习和深度学习的算法，但针对TTP的研究有限。在本文中，我们提出了第一个基于强化学习的多起点邻域搜索算法，表示为钢筋cens，用于求解TTP。为了加快搜索速度，我们采用了邻域约简的预处理程序。利用TSP路由和迭代贪婪包装分别构建高质量的初始解，并通过基于强化学习的邻域搜索进一步改进。此外，设计了一个后优化程序，以持续改进解决方案。我们对60个常用的基准实例进行了广泛的实验，涉及文献中的76到33810个城市。实验结果表明，在相同的时间限制下，本文提出的算法在求解质量方面优于三种最先进的算法。特别是，在最近的TTP竞赛中公开报道的18个实例中，ReinforceNS获得了12个新结果。我们还进行了一个额外的实验来验证强化学习策略的有效性。

## 介绍

旅行贼问题（TTP）是TSP和背包（KP）的组合。

第一个引入强化学习解决TTP的多启动邻域搜索算法。据该系统集成了预处理、初始化、基于强化学习的邻域搜索和后优化。为了确保搜索空间的有效探索，reinforcement采用了多启动机制来避免局部最优陷阱，并采用了专用初始化来生成新的高质量初始解。

贡献：

- 提出的算法集成了预处理过程和基于强化学习的邻域搜索过程，以确保对有希望的邻域解进行有效和高效的检查。
- 应用后优化程序进一步改进局部最优解。这是由项目翻转阶段和路由精炼阶段共同保证的。
- 对常用的60个TTP基准实例进行了广泛的测试。实验结果表明，该算法在相同的时间内优于现有的三种算法。特别是，在最近的TTP竞赛中公开报道的18个案例中，它发现了12个改进的结果。

## 相关工作

- TTP

  - 精确算法：动态规划、混合整数规划
  - 元启发式：遗传规划生成低级启发式算法来探索路线或者包装计划
  - 构造启发式：链式LKH（CLKH）生成路线，在保持该路线不变的情况下设计贪婪包装策略
  - 联合启发式：在通过构造启发式生成初始TTP解后，迭代地应用TSP和KP两个单独的搜索程序来改进路线和背包路线
  - Full-encoding：不分离TSP和KP，采用混合进化框架

  实验中采用了后三种方案中的SOTA：S5、CoCo (Cooperation Coordination)、MATLS (Memetic Algorithm with Two-Stage Local Search)

## 问题描述

就是TSP+背包，即每个城市有若干个物品，每个物品有价值 $p$ 和重量 $w$ ，小偷有一个背包，在旅行期间，小偷可以在任何城市将任何物品装入背包，直到物品的总重量不超过背包的最大重量W。背包的重量与小偷的移动速度有一个线性关系。

解表示为 $(\Pi,P)$，其中 $\Pi=(x\_1,x\_2,\cdots,x\_n)$ 表示路径，$P=(z\_1,\cdots,z\_m)$ 表示取了哪些物品，$z\_i$ 是二元变量。

## ReinforceNS算法

本文提出的ReinforceNS算法是一种基于强化学习机制的多起点迭代邻域搜索启发式算法。预处理后，通过重复构造专用解、局部优化和后优化来探索搜索空间。局部优化和后优化过程通过从当前解过渡到它的一个邻近解来应用邻域搜索。每次迭代都涉及到将当前解决方案替换为对当前解决方案提供即时改进的最佳邻近解决方案。

<img src="{{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024115902748.png" alt="image-20241024115902748" style="zoom:50%;" />

算法1提供了基于多启动迭代局部搜索框架的一般方案。

具体来说，在初始化最优解 $(\Pi^\ast,P^\ast)$ 及其相应的目标函数值 $f^\ast$之后，ReinforceNS 预处理程序减少候选边缘集，然后执行一系列迭代，直到达到最大时间限制 $T_{max}$。每次迭代时，TSP求解器CLKH生成一个解 $\Pi$。随后，对 $\Pi$ 应用迭代贪婪装箱过程（IGP），生成装箱计划 $P$，从而初始化 TTP 的高质量解 $(\Pi,P)$ 。

该方案通过基于强化学习的邻域搜索过程得到改进。为了使搜索摆脱局部最优，采用了包含物品翻转阶段和路径精炼阶段的后优化过程进行进一步改进。最优解（Π∗,P∗）被更新并最终作为输出返回到 ReinforceNS 算法的末尾。

### 预处理

在开始搜索之前，使用邻域约简过程预处理输入TTP实例 $G$。通过采用 LKH 中的1树和 $\alpha$ -邻近的子梯度优化方法来实现。

给定一个完整图 $G$，城市 $x\_i$ 的邻域 $\mathcal{N}(x\_i)$ 定义为与 $x\_i$ 通过边 $\left(x\_i, x\_j\right)$ 相连的相邻城市 $x\_j$ 的集合。最初，每个顶点有 $n-1$ 个邻居。

$G$ 的1树是一个覆盖顶点集 $V \backslash x\_i$ 的生成树，并结合与顶点 $x\_i$ 相连的两条边，这里 $x\_i$ 是 $V$ 中的一个任意特殊顶点。最小1树 $T$ 是长度最小的1树。边 $\left(x\_i, x\_j\right) \in E$ 的 $\alpha$ -邻近定义为 $\alpha(i, j)=L\left(T^{+}\left(x\_i, x\_j\right)\right)-L(T)$，其中 $L(T)$ 是 $T$ 的长度，$L\left(T^{+}\left(x\_i, x\_j\right)\right)$ 是包括边 $\left(x\_i, x\_j\right)$ 的最小1树的长度。

为了减少每个顶点的邻域，我们使用子梯度优化程序，通过对每个顶点引入惩罚，逐渐将顶点的度数调整为2，以获得最小1树。在我们的实验中，$\mathcal{N}\left(x\_i\right)$ 的大小设置为15。ReinforceNS算法利用这些减少的邻域进行后续搜索。

### 初始化

为了确保高效的搜索，初始解的质量对于引导搜索到有前景的区域至关重要。分别应用TSP和KP方法生成一个高质量的初始TTP解 $(\Pi, P)$。

采用 CLKH 生成高质量的路线 $\Pi$ （CLKH是解决TSP的最广泛使用算法之一）。

基于路线 $\Pi$，设计了一个专门的迭代贪婪背包（IGP）程序，以获得良好的背包计划 $P$，从而构建初始解 $(\Pi, P)$。采用以下公式作为评分函数来评估从城市 $x_i$ 拾取的物品 $k$。

$$
s_{i_k}=\frac{p_{i_k}}{w_{i_k} \times \xi_i}
$$

其中，$\xi\_i=d\_{x\_n, x\_1}+\sum\_{j=i}^{n-1} d\_{x\_j, x\_{j+1}}$ 表示城市 $x\_i$ 到路线终点城市（仓库）的距离。注意到在重量和利润值之间存在权衡，还在公式中引入了指数，因此函数变为以下公式。

$$
s_{i_k}=\frac{p_{i_k}^\kappa}{w_{i_k}^\beta \times \xi_i^\gamma}
$$

使用三个参数 $\kappa, \beta$ 和 $\gamma$ 分别调整利润值、重量和距离的影响。

IGP 由四个步骤组成。

- 首先，从0到1的均匀分布中随机抽取 $\kappa, \beta$ 和 $\gamma$ 的值。
- 第二，迭代选择通过上式计算出的得分最高的物品，并将其装入背包。当物品的总重量不超过背包最大重量 $W$ 乘以随机选择的因子 $\delta$ （在0.5和1之间），则获得一个可行的背包 $P_0$ 。
- 第三，重复上述两个步骤 $\frac{10000}{\log (n \cdot m)}$ 次，以确定最佳背包 $P_0^{\prime}$ 。
- 最后，从获得 $P_0^{\prime}$ 的参数 $\kappa, \beta, \gamma$ 和 $\delta$ 开始，进一步通过调整 $\kappa$ 和 $\beta$ 来优化背包方案的搜索，如 $(\kappa-\theta, \beta-\theta), (\kappa+\theta, \beta+\theta), (\kappa-\theta, \beta+\theta)$ 和 $(\kappa+\theta, \beta-\theta)$ ，其中 $\theta$ 设为5，并在搜索过程中以0.75倍逐渐减少。该过程重复20次，最终得到一个高质量的初始背包解 $P$。因此，初始解 $(\Pi, P)$ 得以有效构建。

### 基于强化学习的邻域搜索

邻域搜索是一种强大的局部优化方法，应用强化学习机制加快邻域检查，并提出了基于强化学习的邻域搜索（RLNS），以发现更优解。

回顾 $L(T)$ 是最小1树的长度，边 $\left(x\_i, x\_j\right)$ 的 $\alpha$ -邻近值 $\alpha(i, j)$ 是为了包含这条边而增加的 $L(T)$ 长度。定义城市 $x\_i$ 和城市 $x\_j$ 的Q值，其中参数 $\tau$ 设为0.5，$b$ 设为 $1e^{-6}$。

$$
Q(i, j)=\frac{L(T)}{\tau \cdot \alpha(i, j)+(1-\tau) \cdot d_{i, j}+b}
$$

RLNS伪代码如算法2所示。

<img src="{{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024194200406.png" alt="image-20241024194200406" style="zoom:50%;" />在使用公式(6)初始化每条边的Q值后，RLNS进行一系列迭代，最大迭代次数 $\text{Iter}\_{\text{max}}$ 设为1000。在每次迭代中，RLNS对每个城市 $x\_i$（不包括仓库）使用 $\epsilon$ -贪婪策略选择用于2-opt交换的邻近城市 $x\_j$。更准确地说，它以 $1-\epsilon$ 的概率从邻域 $\mathcal{N}(x\_i)$ 中选择Q值最高的邻近城市 $x\_j$，并以 $\epsilon=0.2$ 的概率随机选择 $x\_j$。注意，因 $x\_i$ 到 $x\_j$ 的子路径发生变化，来自这些城市的物品应从 $P$ 的背包中移除。对于沿着从 $x\_j$ 到 $x\_i$ 的新子路径的城市，RLNS从这些城市中选择得分最高的物品，将其加入背包，直到总重量超过背包最大重量 $W$ 的 $\delta$ 倍为止。一旦发现更优的目标值，局部最优解将被更新。随后，使用下式更新 $\left(x\_i, x\_j\right)$ 的Q值。此处，参数 $\lambda$ 和 $\phi$ 分别设为0.1和0.9，状态 $s\_t$、动作 $a\_t$ 和奖励 $r(s\_t, a\_t)$ 定义如下：

$$
\begin{aligned}
Q(s_t, a_t)= & (1-\lambda) \cdot Q(s_t, a_t) + \lambda \cdot [r(s_t, a_t) + \phi \max_{a'} Q(s_{t+1}, a')]
\end{aligned}
$$

- **状态 $s\_t$**：当前迭代 $t$ 的状态是城市 $x\_i$，它将选择一个邻近城市进行2-opt交换操作。
- **动作 $a\_t$**：在当前迭代 $t$ 选择城市 $x\_i$ 的一个邻近城市 $x\_j$。
- **转换**：执行该动作后的下一个状态是需要为2-opt交换操作选择邻近城市的下一个城市。

- **奖励 $r\left(s\_t, a\_t\right)$**：假设当前解为 $\left(\Pi\_0, P\_0\right)$ ，在城市 $x\_i$ 和 $x\_j$ 进行2-opt交换操作后，得到的邻域解为 $\left(\Pi\_1, P\_1\right)$。邻域解与当前解的目标值差异表示为 $r\left(s\_t, a\_t\right)=f\left(\Pi\_1, P\_1\right)-f\left(\Pi\_0, P\_0\right)$。

在连续的 $\text{Iter}\_{\text {max }}$ 次迭代中，如果找到更优解，则更新最佳解 $\left(\Pi^{\prime}, P^{\prime}\right)$，并将其作为RLNS的最终输出返回。

### 后期优化过程

除了上述的RLNS过程，提出的 ReinforceNS 算法还开发了一个后期优化过程，以进一步提高通过RLNS获得的最佳解 $\left(\Pi^{\prime}, P^{\prime}\right)$ 的质量。该后期优化过程由物品翻转阶段和路径提炼阶段共同完成。

物品翻转阶段旨在优化背包解 $P^{\prime}$，同时保持路径解 $\Pi^{\prime}$ 不变。物品的翻转操作包括从背包中移除已包含的物品，或将未包含的物品加入背包。在每次迭代中，物品翻转阶段对每个物品 $i$ 伪执行翻转操作。如果该操作产生了新的目标值，则使用物品翻转操作翻转物品 $i$。如果没有取得改进，物品 $i$ 保持不变。该过程重复进行，直到某次迭代中不再发现新的目标值为止。

从 TTP 的问题定义可以很容易地注意到，当盗贼接近终点城市（仓库）时，背包的重量增加，导致速度下降。我们引入了一个路径优化阶段，尝试将包含重但有价值物品的城市移到路径的最后位置，以进一步优化 TTP 解。

<img src="{{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024195117797.png" alt="image-20241024195117797" style="zoom:50%;" />

图 1 提供了一个示例，说明了在调整路径中城市的位置时，如何有效计算时间成本 $\psi(\Pi, P)$。假设路径解为 $\Pi$，背包解为 $P$，边 $\left(x\_i, x\_{i+1}\right),\left(x\_{j-1}, x\_j\right)$ 和 $\left(x\_j, x\_{j+1}\right)$ 的长度分别为 7、4 和 3，盗贼在城市 $x\_i$ 的速度为 5，在城市 $x\_{j-1}$ 的速度为 2。城市 $x\_j$ 被移动到路径中城市 $x\_i$ 和 $x\_{i+1}$ 之间，形成新路径 $\Pi^{\prime}$。新边 $\left(x\_i, x\_j\right),\left(x\_j, x\_{i+1}\right)$ 和 $\left(x\_j, x\_{j+1}\right)$ 的长度分别为 5、6 和 5。由于城市 $x\_j$ 没有被拾取的物品，盗贼的速度保持不变。使用公式，时间增量 $\triangle \psi$ 计算为 $\frac{5+6}{5}+\frac{5}{2}-\left(\frac{7}{5}+\frac{3+4}{2}\right)=-0.2$，其可以在 $O(1)$ 时间复杂度内有效计算。可以看到，调整路径中城市的位置使新路径 $\Pi^{\prime}$ 更长（16 比 14），但减少了时间成本（-0.2），从而得到一个更优的路径解。

$$
\begin{aligned}
\Delta \psi & =\psi\left(\Pi^{\prime}, P\right)-\psi(\Pi, P) \\ 
& =\frac{d_{x_i, x_j}+d_{x_j, x_{i+1}}}{v_{x_i}}+\frac{d_{x_{j-1}, x_{j+1}}}{v_{x_{j-1}}} \\
& -\left(\frac{d_{x_i, x_{i+1}}}{v_{x_i}}+\frac{d_{x_{j-1}, x_j}+d_{x_j, x_{j+1}}}{v_{x_{j-1}}}\right)
\end{aligned}
$$

为了更清楚地说明路径优化阶段，路径 $\Pi\_x$ 被定义为从仓库开始并结束于城市 $x$ 的路线。对于每个不包含在背包中的城市 $x\_i$，我们伪插入它到 $\Pi\_{x\_i}$ 的每条边中，并使用上式有效计算时间成本。如果发现多个改进的目标值，城市 $x\_i$ 将被插入到具有最佳目标值的边中。如果这些城市中没有任何改进的目标值，路径优化阶段将停止。

## 实验

与 GECCO 2023 会议上 TTP 竞赛的最佳公开报告结果进行比较。

![image-20241024195446547]({{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024195446547.png)

从 benchmark 中的三类 CatA（背包的最大重量相对较小。每个城市只有一件物品。项目的权重和利润是有界且强相关的。）、CatB（背包的最大重量适中。每个城市有五件物品。项目的权重和利润是不相关的，所有项目的权重都是相似的。）、CatC（背包最大重量高。每个城市有10个项目。项目的权重和利润是不相关的。）各抽了 10 个实例。

![image-20241024195502965]({{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024195502965.png)

对各个算法统计了基于最佳平均结果的95%置信区间

<img src="{{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024200621526.png" alt="image-20241024200621526" style="zoom:50%;" />

消融实验对是否使用强化学习进行的消融：

![image-20241024200651019]({{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024200651019.png)

也统计了基于最佳平均结果的95%置信区间

<img src="{{site.url}}/img/2024-10-24-ReinforceNS-Reinforcement_Learning-based_Multi-start_Neighborhood_Search_for_Solving_the_Traveling_Thief_Problem/image-20241024200758475.png" alt="image-20241024200758475" style="zoom:50%;" />

（感觉强化学习只是锦上添花）

## 总结

未来工作：用其他强化学习或者深度学习技术。



