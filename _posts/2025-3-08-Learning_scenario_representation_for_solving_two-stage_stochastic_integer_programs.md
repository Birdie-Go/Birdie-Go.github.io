---
layout:     post
title:      Learning scenario representation for solving two-stage stochastic integer programs
subtitle:   ICLR2022 二阶段随机优化 学习情景重建
date:       2025/03/08
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - ICLR
---

Learning scenario representation for solving two-stage stochastic integer programs

ICLR 2022

作者：Yaoxin WU, Wen SONG, Zhiguang CAO, Jie ZHANG

## 省流

文章提出了一种基于条件变分自编码器（Conditional Variational Autoencoder, CVAE）的方法，用于学习SIPs中情景的表示（representations）。这种方法的核心思想是将每个情景及其对应的确定性部分（上下文）嵌入到一个低维的潜在空间中。通过这种方式，可以有效地捕捉情景与其实例之间的依赖关系。

具体来说，作者设计了一个GCN的编码器，用于将情景和上下文嵌入到潜在空间中。然后，解码器根据上下文从潜在表示中重建情景。这种设计确保了学习到的情景表示能够正确地与其对应的实例相关联。

此外，文章还将学习到的情景表示应用于两个典型任务：情景缩减和目标预测。通过聚类算法找到代表性情景，从而降低问题复杂度，并获得高质量的近似解。同时，通过半监督学习扩展编码器，预测每个情景的目标值，进一步缩小与原始SIPs最优解的近似误差。

（即，SIP通过蒙特卡洛采样多个情景来讲SIP转换成MIP，而本文通过一个网络，输出一些具有代表性更有利于解决问题的情景。）

## 前序知识

### 两阶段随机整数规划

（省流：可以通过蒙特卡洛采样多个随机分布的场景将SIP转换成MIP）

作为一种优化问题，随机整数规划（SIP）通常由不确定参数（可能遵循某种概率分布）和由于整数约束导致的离散解空间所定义。它通常以两阶段形式描述如下：

$$
\min_{x} \mu^\top x + \mathbb{E}_\omega[Q(x, \omega)] \quad (1)
$$

满足：

$$
Ax \leq b, \quad x \in \mathbb{R}^{n_1 - p_1} \times \mathbb{Z}^{p_1}, \quad (2)
$$

其中，$Q(x, \omega) := \min_y \lbrace q_\omega^\top y \mid W_\omega y \leq h_\omega - T_\omega x; y \in \mathbb{R}^{n_2 - p_2} \times \mathbb{Z}^{p_2}\rbrace $。

具体来说，式(1)和(2)定义了第一阶段问题，其中$x \in \mathbb{R}^{n\_1}$是决策变量，$n\_1 \geq p\_1$；$Q(x, \omega)$定义了第二阶段问题，其中$y \in \mathbb{R}^{n\_2}$是决策变量，$n\_2 \geq p\_2$。我们称第一阶段中的静态参数组$\mu \in \mathbb{R}^{n\_1}$、$A \in \mathbb{R}^{m\_1 \times n\_1}$和$b \in \mathbb{R}^{m\_1}$为上下文（context），并假设不确定参数组$q\_\omega \in \mathbb{R}^{n\_2}$、$W\_\omega \in \mathbb{R}^{m\_2 \times n\_2}$、$T\_\omega \in \mathbb{R}^{m\_2 \times n\_1}$和$h\_\omega \in \mathbb{R}^{m\_2}$遵循分布$P$。本文专注于基于图的SIPs，这是一类在许多领域（如网络、交通和调度）中具有实际应用的问题。

解决上述SIP的主要方法是样本平均近似，它**通过蒙特卡洛模拟将SIP转换为混合整数规划**（MIP），并优化以下目标：

$$
O(x) := \min_x \mu^\top x + \frac{1}{N} \sum_{i=1}^N Q(x, \omega_i), \quad (3)
$$

其中，$\lbrace \omega\_i\rbrace \_{i=1}^N$是从$P$中抽取的一组情景，即不确定参数的独立同分布（i.i.d.）样本。通常，为了使情景分布$\hat{P}$与$P$拟合良好，需要大量的情景集，这可能导致给定求解器的MIP难以处理。在本文中，我们的目标是通过学习情景的表示来找到$\lbrace \omega\_i\rbrace \_{i=1}^N$中少量的信息量大的代表性情景，从而显著提高计算效率，同时容忍可接受的近似误差。

### 条件变分自编码器

作为一种无监督生成模型，条件变分自编码器（CVAE）是基于变分自编码器（VAE）发展而来的，并且它进一步**通过附加的随机变量来控制数据生成过程**。这些条件变量可以是类别标签，或者是具有特定分布的数据属性，它们被输入到编码器和解码器中。在本文中，我们利用CVAE来学习SIPs中情景的表示，条件变量是从上下文中的连续参数中派生出来的。CVAE在输入数据的边际似然上的证据下界目标（ELBO）可以表示为：

$$
\log p_\theta(X, c) \geq \text{ELBO}(X, c) = \mathbb{E}_{q_\phi(z|X,c)}[\log p_\theta(X|z, c)] - \text{KL}[q_\phi(z|X, c)\parallel p(z|c)],
$$

其中，$X$、$c$和$z$分别表示输入、潜在变量和条件变量；编码器$q\_\phi$和解码器$p\_\theta$通常由深度神经网络参数化。与大多数现有工作中条件变量的简单分布不同，本文利用CVAE来学习SIPs中情景的表示，条件变量是从上下文中的连续参数中派生出来的。

## 方法论

给定一类随机整数规划（SIP）实例 $\lbrace X\_m\rbrace \_{m=1}^M$，其参数从分布$D$中抽取，我们将每个实例视为一个2元组$X\_m = (D\_m, \lbrace \omega\_i^m\rbrace \_{i=1}^N)$，其中$D\_m$表示第$m$个实例中的上下文（即静态参数组），$\omega\_i^m$表示第$m$个实例中的第$i$个情景（即不确定参数的第$i$个实现）。我们的目标是学习每个实例中情景$\lbrace \omega\_i^m\rbrace \_{i=1}^N$在上下文$D\_m$下的潜在表示（变量）$\lbrace z\_i^m\rbrace \_{i=1}^N$。

### 用于情景表示的条件变分自编码器（CVAE）

我们的CVAE包含一个用于推断过程的编码器和一个用于生成过程的解码器，它们分别由深度神经网络$q\_\phi$和$p\_\theta$参数化。在生成过程中，解码器近似给定潜在变量和条件变量的情景中不确定参数的后验分布，表示为：

$$
p_\theta(\omega_i^m \mid z_i^m, c^m) = f(\omega_i^m; z_i^m, c^m, \theta); \quad q_\phi(c^m \mid D_m) = h(c^m; D_m, \phi), \quad (5)
$$

其中，$f$和$h$表示似然函数（例如高斯分布或多项分布），分别由$p\_\theta$和$q\_\phi$实例化。需要注意的是，我们不直接将上下文$D\_m$的原始数据作为解码器的条件变量，因为它们是连续且高维的。这不仅会增加计算复杂度，还可能淹没通常是低维的潜在变量。相反，我们利用图神经网络（GNN）在编码器中从$D\_m$中导出一个低维的条件变量$c^m$。此外，通过保持潜在变量和条件变量之间的边际独立性，解码器将更有效地根据上下文$D\_m$生成$\omega\_i^m$。

在推断过程中，编码器近似给定$(\omega_i^m, D_m)$的情景潜在表示的后验分布，表示为：

$$
q_\phi(z_i^m \mid \omega_i^m, D_m) = \mathcal{N}(z_i^m \mid \mu_\phi(\omega_i^m, D_m), \sigma_\phi^2(\omega_i^m, D_m)), \quad (6)
$$

其中，我们假设情景表示服从高斯分布，均值$\mu\_\phi$和标准差$\sigma\_\phi$分别由神经网络参数化。如前所述，我们利用图神经网络（GNN）在编码器中嵌入高维且连续的$\omega\_i^m$和$D\_m$，同样也用于解码器中嵌入单独的$D\_m$。这种参数共享设计有助于快速学习嵌入，从而导出潜在变量和条件变量，即$z\_i^m$和$c^m$。为了学习这两种变量的独立表示，我们强制执行条件独立性：

$$
q_\phi(z_i^m, c^m \mid \omega_i^m, D_m) = q_\phi(z_i^m \mid \omega_i^m, D_m) q_\phi(c^m \mid D_m). \quad (7)
$$

总结来说，我们的CVAE依赖于两个重要步骤：

1）我们首先使用编码器获得每个SIP实例中情景相对于上下文的表示；

2）然后我们使用解码器根据上下文从潜在空间重建情景。

**半监督CVAE**。通过推断出的情景表示，我们可以直接为解决SIP的多种下游任务提供支持，例如使用现成的聚类算法进行情景缩减。然而，这种方法可能会忽略情景与目标值之间的关系，这在实践中是不可忽视的，因为即使相似的情景也可能产生不同的解决方案。因此，我们还将CVAE扩展到以半监督的方式预测目标函数。鉴于情景已被嵌入到连续空间中，预测过程预计将具有良好的泛化能力。

大多数用于半监督学习的CVAE模型预测离散目标，并将真实值（例如类别标签）视为条件变量。相比之下，我们预测数据的连续属性（即目标值），并直接通过编码器对其进行近似。为此，我们使用CPLEX求解器仅针对少量实例（1%的$M$）收集由相应情景和上下文定义的MIP问题的最优目标值$\lbrace Y\_i^m\rbrace \_{i=1}^N$。我们将底层的$(\omega\_i^m, Y\_i^m, D\_m)$联合分布记为$D\_Y$。然后，我们使用一个子网络处理编码器中的GNN嵌入$(\omega\_i^m, D\_m)$，目标函数$\sigma$参数化为：

$$
r_\psi(Y_i^m \mid h_i^m) = \sigma(Y_i^m; h_i^m, \psi); \quad q_\phi(h_i^m \mid \omega_i^m, D_m) = g(h_i^m; \omega_i^m, D_m, \phi), \quad (8)
$$

其中，$h\_i^m$表示由$q\_\phi$导出的GNN嵌入。

### 神经网络

为SIPs参数化CVAE的一个主要挑战是如何有效地嵌入上下文，因为它具有高维且连续的特点。此外，它还涉及编码器和解码器的输入，并显著影响最终的情景表示。直接使用多层感知机（MLP）处理$D\_m$中的静态参数可能会失败，因为它无法利用问题结构，并且无法泛化到不同大小的实例。为了克服这些限制，我们利用图神经网络（GNN）嵌入上下文$D\_m$，以获得解码器中的条件变量。同时，我们应用相同的GNN嵌入每个情景及其上下文$(\omega\_i^m, D\_m)$，以获得编码器中的潜在表示。为此，我们首先在图上描述上下文和情景。

**SIP图**。我们定义一个完全图$G = (V, E)$，其中$V = \lbrace v\_1, \dots, v\_n\rbrace $表示节点，其特征为$V \in \mathbb{R}^{n \times d\_v}$；$E = \lbrace e\_{jk} \mid v\_j, v\_k \in V\rbrace $表示边，其特征为$E \in \mathbb{R}^{n \times n \times d\_e}$。具体来说，第$j$个节点上的特征表示为$v\_j^{\omega\_i} = [s\_j^{\omega\_i}; d\_j]$（$[;]$表示连接操作）。其中，$s\_j^{\omega\_i} \in \omega\_i^m$表示第$j$个节点上的不确定参数的实现，$d\_j \in D\_m$表示第$j$个节点上的静态参数。类似地，边$e\_{jk}$上的特征表示为$e\_{jk}^{\omega\_i} = [s\_{jk}^{\omega\_i}; d\_{jk}]$，其中$s\_{jk}^{\omega\_i}$和$d\_{jk}$分别表示来自第$i$个情景和上下文的参数。与现有工作只考虑节点上的变化参数不同，我们的图表示更为通用，可以应用于更广泛的SIPs类别。

在SIP图的基础上，我们可以利用GNN分别为上下文$D\_m$或变化的情景与上下文$(\omega\_i^m, D\_m)$（$i = \lbrace 1, \dots, N\rbrace $）导出嵌入，分别记为$h\_m$和$h\_i^m$。在本文中，我们采用了图卷积网络（GCN），这是GNN的一个常用变体。

**编码器**。编码器处理图嵌入$h\_i^m$，通过两个独立的线性投影分别计算潜在变量的均值和标准差的二维向量，即式(6)中的$\mu\_\phi(\omega\_i^m, D\_m)$和$\sigma\_\phi(\omega\_i^m, D\_m)$。同样，我们通过线性投影将图嵌入$h\_m$映射为一个二维向量，即式(5)中的$c^m$。

**解码器**。我们将$c^m$与潜在变量$z_i^m$连接起来，并通过一个包含两个隐藏层的MLP来重建情景。隐藏层的维度分别为128和256，激活函数为ReLU。

**半监督学习**。为了实现半监督CVAE的功能，我们利用另一个MLP作为子网络来处理$h\_i^m$。该子网络包含一个512维的隐藏层，激活函数为ReLU，并输出一个单一值以估计目标值$Y\_i^m$。

![image-20250308155051491]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308155051491.png)

基本的神经网络结构如图1所示，其中$\hat{\omega}\_i^m$和$\hat{Y}\_i^m$分别是重建的情景和预测的目标值。需要注意的是，我们通过在上下文中填充0来嵌入单独的上下文（灰色方块中的部分）。这样可以共享GCN来同时获得$h\_m$和$h\_i^m$的嵌入。

### 训练与推理

![image-20250308155152248]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308155152248.png)

训练目标是最小化三个损失函数。第一个损失是重建情景与原始情景之间的均方误差（MSE），它在CVAE中用于最大化数据的边际似然。第二个损失是潜在变量与先验高斯分布$p(z) = \mathcal{N}(z \mid 0, I)$之间的KL散度，用于正则化。第三个损失是预测目标值与真实值之间的MSE。正式地，训练CVAE的目标可以表示为：

$$
L(\theta, \phi, \psi) = \mathbb{E}_D[-\text{ELBO}(X_m)] - \alpha \cdot \mathbb{E}_{D_Y}[\log r_\psi(Y_i^m \mid h_i^m)],
$$

其中，$\text{ELBO}(X\_m) = \mathbb{E}\_{q\_\phi(z\_i^m \mid \omega\_i^m, D\_m)}[\log p\_\theta(\omega\_i^m \mid z\_i^m, c^m)] - \beta \cdot \text{KL}[q\_\phi(z\_i^m \mid \omega\_i^m, D\_m) \parallel p(z)]$ 包含了前两个MSE和KL散度；$\alpha$和$\beta$是平衡三个损失的超参数。按照VAE的典型训练范式，我们通过重参数化技巧和蒙特卡洛近似联合优化$\theta$和$\phi$。训练过程总结在算法1中，我们使用Adam优化器更新参数。

如算法1所示，我们在每个实例中随机选择一个情景，以便能够以批量方式处理它及其上下文。这种跨不同实例的上下文和情景的泛化将允许快速训练神经网络。然而，在推理过程中，我们使用训练好的编码器来获得每个实例中的潜在表示，通过并行处理所有情景来实现。这是通过在每个实例中为所有情景复制上下文来完成的。通过这种方式，可以显著提高解决具有大量情景的实例的计算效率，从而处理大量实例。对于每个实例，从编码器中导出的情景表示随后用于聚类，以找到原始情景中的代表性中心，从而减少情景数量。

### 应用

![image-20250308155218558]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308155218558.png)

我们在两类典型的SIP问题中应用我们的方法，即网络设计问题（NDP）和设施选址问题（FLP））。我们将它们表示为SIP图，其节点和边上的特征（包括静态和不确定参数）如表1所示。对于FLP，用F.和C.标记的参数分别表示与设施和客户节点相关的参数。

## 实验

问题：

- 网络设计问题（NDP）
- 设施选址问题（FLP）

训练CVAE仅学习情景的表示（算法1中的第5-8行），训练好的编码器可以在每个实例中获得表示，这些表示直接用于通过聚类算法找到代表性情景，从而进行情景缩减。另一方面，我们训练半监督CVAE同时学习情景表示和预测目标值（算法1中的第5-13行）。训练完成后，神经网络可以为每个情景获得表示和目标值，我们将它们连接起来用于聚类，以找到更具信息量的代表性情景。我们将上述两种范式分别称为CVAE-SIP和CVAE-SIPA。

实例：

- NDP：14个节点（2个源节点、2个目标节点和10个中间节点）
- FLP：10个设施节点和20个客户节点
- 每个实例生成200个情景
- 泛化：加倍和四倍增加中间节点数量来生成更大规模的NDP实例；情景数量也加倍和四倍操作

训练：

- 徐联机：12800，归一化到0-1之间
- 两个问题分别训练 100 和 400 轮
- 1 台 2080Ti

baseline：

- CPLEX
- K-medoids：与本文方法中使用的相同聚类算法
- Scenario-M：一种监督方法，使用手工设计的特征来预测用于解决FLP变体的情景
- Solution-M：一种监督方法，使用原始特征预测用于解决血液运输问题的第一阶段决策变量

![image-20250308155956486]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308155956486.png)

## 附录

- GCN的设计

- NDP的定义

- FLP的定义

- 实例生成的细节

- 训练的细节

- 大规模FLP的补充实验

  ![image-20250308160219826]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160219826.png)

- 不同分布和不同依赖关系的泛化

  ![image-20250308160314087]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160314087.png)

- 潜在表征的客观预测

  ![image-20250308160356195]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160356195.png)

  ![image-20250308160403789]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160403789.png)

- 和OR的方法比较

  ![image-20250308160424053]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160424053.png)

- 消融实验

  ![image-20250308160445449]({{site.url}}/img/2025-3-08-Learning_scenario_representation_for_solving_two-stage_stochastic_integer_programs/image-20250308160445449.png)







