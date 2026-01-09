---
layout:     post
title:      Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed
subtitle:   AAAI2024 引导知识蒸馏 非自回归模型
date:       2025/04/28
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - AAAI
---

Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed

AAAI2024

![image-20250428204454556]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428204454556.png)

## 省流

就看方法论里面的那张图，做了一个从自回归引导到非自回归的知识蒸馏。

## 摘要

神经网络结构模型采用自回归（AR）或非自回归（NAR）学习方法，在车辆路径问题（VRP）中表现出了良好的性能。虽然AR模型产生高质量的解决方案，但由于其顺序生成的性质，它们通常具有较高的推理延迟。相反，NAR模型以低推理延迟并行生成解决方案，但通常表现出较差的性能。我们提出了一种通用的指导非自回归知识蒸馏（GNARKD）方法，以获得高，GNARKD消除了AR模型中顺序生成的约束，同时保留了网络架构中学习到的关键组件，通过知识蒸馏得到相应的NAR模型，并将其应用于三种广泛采用的AR模型，以获得合成和真实世界实例的NAR VRP求解器。

实验结果表明，GNARKD在保证性能下降（2 ~ 3%）的前提下，显著缩短了推理时间（4 ~ 5倍）。据我们所知，本研究首次通过知识蒸馏从AR求解器获得NAR VRP求解器。



## 引入

贡献

- 发现NAR模型在求解VRP时的表现不佳可以归因于它们在推理过程中倾向于采取不太自信的行动，该工作是第一个在VRP领域报告这一发现的实验结果的支持。
- 提出了一种新的通用方法GNARKD，将AR模型转换为NAR模型，在保持基本知识的同时提高了推理速度。GNARKD是第一个通过KD从AR模型获得NAR VRP求解器的方法。
- 将GNARKD应用于三种广泛采用的AR模型，并使用广泛使用的合成数据集和真实世界数据集评估其性能。实验结果表明，GNARKD与教师模型相比，显著减少了推理时间，并获得了同等的解决方案质量。这一发现表明，衍生的NAR模型适合部署在需要瞬时，接近最优解的真实世界场景中。

## 引导NAR知识蒸馏

![image-20250428204731709]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428204731709.png)

### 学生模型的架构

解码器只包括前馈层和多头注意模块，没有递归结构。

**编码器**

不变，为了促进知识从教师到学生的转换。

**解码器**

（省流，就解码器的输入改成了所有节点embedding的拼接，以及一个可学习的参数）

所有三个关键增强都围绕解码器展开，如下所述：

1. **解码器的输入和输出**：在自回归（AR）模型中，解码器负责根据节点表示和先前选择节点的操作历史生成用于选择后续节点的操作概率分布。这一过程通常描述如下：

   $$
   \lambda_{te}^{t} = \sigma_{T1} \left( f_{te} \left( z, \text{if } t = 1, \text{CE}(a_{1:t-1}), \text{if } t > 1, h_{te} \right) \right)
   $$

   其中，$\lambda\_{te}^{t} \in \mathbb{R}^{1 \times n}$，$\sigma\_{T1}$ 表示温度为 $T1$ 的 Softmax 函数，$f\_{te}$ 表示教师解码器的前向传播函数，$z \in \mathbb{R}^{d\_h \times 1}$ 表示一个可学习的参数，作为输入占位符（或 CVRP 中的配送中心），$\text{CE}(a\_{1:t-1})$ 表示基于节点选择操作历史的上下文嵌入。不同的 AR 模型可能使用不同的方法计算上下文嵌入，但目的都是构建一个查询 $q\_t \in \mathbb{R}^{d\_h \times 1}$，用于选择时间 $t$ 的节点。为了获得能够并行传播的学生解码器的非自回归形式，我们省略了操作历史信息的包含，转而建模每个节点与其直接后继节点之间的相关性。此外，为了增强 GNARKD 在不同 GNARKD 学生模型中的通用性，我们采用了一种一致且简单的查询构建方法。具体来说，我们对每个节点嵌入 $h\_{st}^i$ 应用一个共享的线性变换，参数化权重为 $W$，表示为 $\text{emb}\_i = W h\_{st}^i$，$\text{emb}\_i \in \mathbb{R}^{d\_h \times 1}$，作为选择时间 $t$ 节点的查询。随后，解码器通过将所有节点表示进行水平拼接来计算节点 $v\_i$ 连接到另一个节点 $v\_j$ 的紧密度 $A\_{i,j}$，如下所示：

   $$
   A = f_{st} \left( W \left[ z, h_{st}^1, \ldots, h_{st}^n \right] \right), A \in \mathbb{R}^{(n+1) \times n}
   $$

   其中，$\left[ \cdot, \cdot, \cdot \right]$ 表示水平拼接操作符。

2. **位置编码**：由于学生解码器只考虑单个节点而不考虑其在路径中的位置（至少在解码器内是这样），我们避免在学生模型中使用通常在基于 Transformer 的 AR 解码器中使用的位置编码。

3. **未掩蔽注意力**：由于 AR 模型的输出受顺序依赖的限制，其解码器在最终的交叉注意力层中依赖掩蔽以避免重新访问已选择的节点。相反，学生解码器不使用此类掩蔽，因为它仅依赖于解码器后输出处理来生成有效的解决方案。

### 引导知识蒸馏

利用教师模型生成的动作来引导学生模型的解码过程输出，从而使 NAR 学生模型在训练过程中能够学习到 AR 教师模型生成解决方案中的顺序依赖信息。

**引导解码**：要求学生模型复制教师模型生成的相同动作序列 $\pi^{AR}$。然后，对于每个动作 $a\_t \in \pi^{AR}$，使用温度为 $T\_2$ 的 Softmax 函数对选定节点与其他节点 $v\_i$ 之间的连接紧密度进行归一化，从而获得学生在时间 $t$ 的动作概率分布：

$$
\lambda_{st}^t = \sigma_{T2} \left( A_{at,i} \odot \left[ -\infty, \text{if } node \ v_i \in set_t, 1, \text{otherwise} \right] \right)
$$

其中，$i \in \lbrace 1, \ldots, n\rbrace $，$\lambda\_{st}^t \in \mathbb{R}^{1 \times n}$，$\odot$ 表示元素级乘法，$set\_t$ 表示时间 $t$ 的受约束节点集。

**学习代理分布**：我们使用教师模型的动作概率分布作为训练学生的监督信号。为了促进学生通过学习更尖峰的概率分布来增强动作的自信度，我们对教师模型使用温度 $T\_1 < 1$ 的 Softmax 函数，而对学生模型则设置温度 $T\_2 = 1$。具体来说，我们通过最小化学生模型动作概率分布与教师模型之间的 KL 散度来优化学生模型的可学习参数 $\theta$：

$$
\begin{aligned}
L_{KD} &= \mathbb{E}_{\pi^{AR} \sim P^{AR}(\cdot \mid s)} \left[ KL \left[ P^{AR}(\pi^{AR} \mid s) \, \|\, P^{NAR}(\pi^{AR} \mid s, \theta) \right] \right] \\
&= \mathbb{E}_{\pi^{AR} \sim P^{AR}(\cdot \mid s)} \left[ KL \left( \prod_{t=1}^l \lambda_{te}^t \, \Big\| \, \prod_{t=1}^l \lambda_{st}^t \right) \right] \\
&= \frac{1}{B} \sum_{b=1}^B \sum_{t=1}^l \sum_{i=1}^n \lambda_{te}^{t,i} \log \left( \frac{\lambda_{te}^{t,i}}{\lambda_{st}^{t,i}} \right)
\end{aligned}
$$

其中，$B$ 表示训练时使用的批量大小，$\lambda\_{te}^{t,i}$ 和 $\lambda\_{st}^{t,i}$ 分别表示教师和学生在时间 $t$ 选择节点 $v\_i$ 的动作概率。

### 以非自回归方式解决 CVRP 问题

（省流，模型输出热图，通过Greedy的方式获得解）

现提出一种使用 NAR 模型解决有容量约束的车辆路径问题（CVRP）的搜索方法。具体来说，给定 NAR 模型输出的矩阵 A，通过贪心搜索在步骤 t 生成 VRP 解决方案，如下所示：

$$
a_t = \begin{cases}
v_{st}, & \text{if } t = 1, \\
\arg\max(\lambda_{at-1}), & \text{otherwise},
\end{cases}
$$

$$
\lambda_{at} = \sigma_{T2} \left( A_{at,i} \odot \left[ -\infty, \text{if } node \ v_i \in set(t), 1, \text{otherwise} \right] \right),
$$

其中，$i \in \lbrace 1, \ldots, n\rbrace $，$\lambda\_{at} \in \mathbb{R}^{1 \times n}$，而 $v\_{st}$ 表示 VRP 解决方案的起始点。对于 TSP，我们将所有已访问的节点添加到 $set(t)$ 中：

$$
set(t) = \lbrace a_1, a_2, \ldots, a_{t-1}\rbrace .
$$

对于 CVRP，我们在 $set(t)$ 中增加那些需求超过剩余车辆容量的节点，以满足相关约束：

$$
set(t) = \lbrace a_1, a_2, \ldots, a_{t-1}\rbrace  \cup \lbrace  \hat{a}_i, \text{if } \delta_i > Q_t, \emptyset, \text{otherwise} \rbrace ,
$$

其中，$\delta\_i$ 表示节点 $v\_i$ 的需求，$Q\_t$ 表示时间 t 的剩余车辆容量。在时间 t = 1 时，$Q\_1$ 初始化为 $Q\_1 = Q$，之后按以下方式更新：

$$
Q_{t+1} = \begin{cases}
Q, & \text{if } a_t = v_0, \\
Q_t - \delta_i, & \text{otherwise}.
\end{cases}
$$

## 实验

问题

- TSPs和CVRPs

设备

- Intel(R) i5-11400F CPU and an NVIDIA RTX 3060Ti GPU

参数

- 1000 batches of 100 instances in 500/1000 epochs for n = 50/100

学习差距表现

![image-20250428210209283]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210209283.png)

求解质量表现

![image-20250428210424556]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210424556.png)

标准数据集表现

![image-20250428210620326]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210620326.png)

求解速度优势

![image-20250428210555524]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210555524.png)

### 消融实验

对蒸馏温度的敏感性（0.1,0.5,1,5,10）

![image-20250428210732711]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210732711.png)

![image-20250428210858785]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210858785.png)

不同的训练方法（SL，RL，知识蒸馏）

![image-20250428210827076]({{site.url}}/img/2025-4-28-Distilling_Autoregressive_Models_to_Obtain_High-Performance_Non-Autoregressive_Solvers_for_Vehicle_Routing_Problems_with_Faster_Inference_Speed/image-20250428210827076.png)





















