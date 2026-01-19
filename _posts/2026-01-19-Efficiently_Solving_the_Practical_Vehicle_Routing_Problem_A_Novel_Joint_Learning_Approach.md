---
layout:     post
title:      Efficiently Solving the Practical Vehicle Routing Problem - A Novel Joint Learning Approach
subtitle:   看看KDD的NCO
date:       2026/01/19
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - KDD
---

Efficiently Solving the Practical Vehicle Routing Problem: A Novel Joint Learning Approach

蚂蚁集团

KDD2020



## 贡献

本文提出一种基于图卷积网络（GCN）的模型，以节点特征（坐标与需求）和边特征（节点间真实距离）为输入并进行嵌入。我们设计了两个独立的解码器分别对两类嵌入进行解码。一个解码器的输出作为另一个解码器的监督信号。我们提出一种结合强化学习与监督学习的训练策略。在真实世界数据上的综合实验表明：

1. 边特征在模型中被显式考虑时对结果至关重要；
2. 联合学习策略可加速训练收敛并提升解的质量；
3. 我们的模型在文献中的若干著名算法中表现显著更优，尤其当问题规模较大时；
4. 我们的方法对训练时未见过的问题规模具有良好的泛化能力。



## 方法

### 问题设定：图优化视角

一个 VRP 实例可定义在一张图 $\mathbb{G}=(\mathcal{V},\mathcal{E})$ 上，其中  
- 节点集合 $\mathcal{V}=\lbrace 0,\dots ,n\rbrace $，节点 $i=0$ 为仓库（depot），$i\in\lbrace 1,\dots ,n\rbrace $ 为客户；  
- 边集合 $\mathcal{E}=\lbrace e_{ij}\rbrace ,\,i,j\in\mathcal{V}$，表示节点间全部有向边。

仓库节点附带坐标 $x\_{c\_0}$；每个客户节点 $i$ 附带二维特征向量

$$
x_i=\lbrace x_{c_i},x_{d_i}\rbrace ,
$$ 

其中 $x\_{c\_i}$ 为坐标，$x\_{d\_i}$ 为需求量。每条边 $e\_{ij}$ 附带距离 $m\_{ij}$。现实中交通网络不对称，因此 

$$
m_{ij}\neq m_{ji},
$$ 

即图为**全连接有向图**。

给定：  
1. 求解一组环路（routes），每条环路对应一辆车，从 0 出发并返回 0；  
2. 每个客户恰好被访问一次，且车辆载重不超过容量 $c$；  
3. 目标最小化总成本（固定车辆成本 + 行驶成本）。

我们的模型输出一个客户序列 

$$
\pi=(\pi_1,\pi_2,\dots ,\pi_T),\quad \pi_t\in\lbrace 0,1,\dots ,n\rbrace ,
$$ 

其中 0 可出现多次，其余节点仅出现一次。相邻 0 之间的子序即为一条车辆路径。目标函数写作  

$$
\min\; c_v Q_v + c_t \sum_{t=1}^{T-1} m_{\pi_t\pi_{t+1}},
$$ 

其中  

- $c_v$：单辆车固定成本；  
- $Q_v$：使用车辆数；  
- $c_t$：单位距离行驶成本。

### 图卷积网络：节点序列预测与边分类

GCN-NPEC 采用编码器-解码器架构（图 3）。  

![image-20260119161217992]({{site.url}}/img/2026-01-19-Efficiently_Solving_the_Practical_Vehicle_Routing_Problem_A_Novel_Joint_Learning_Approach/image-20260119161217992.png)

- **编码器**：输入节点/边特征，输出节点嵌入 $h\_i^L$ 与边嵌入 $h\_{e\_{ij}}^L$；  
- **解码器 1**（序列预测）：以节点嵌入为输入，输出路径序列 $\pi$；  
- **解码器 2**（边分类）：以边嵌入为输入，输出边“被使用”概率矩阵；  
- **联合训练**：序列解码器结果作为边解码器的伪标签，二者互促。

#### 输入表示

**节点初始特征**（维度 $d_x$）  

$$
x_i=
\begin{cases}
\mathrm{ReLU}\bigl(W_1 x_{c_0}+b_1\bigr), & i=0,\\[4pt]
\mathrm{ReLU}\bigl([W_2 x_{c_i}+b_2;\; W_3 x_{d_i}+b_3]\bigr), & i\ge 1.
\end{cases}
$$

**边初始特征**（维度 $d_y$） 
先构造 k-近邻邻接矩阵（k=10）  

$$
a_{ij}=
\begin{cases}
1,  & j \text{ 是 } i \text{ 的 k-近邻},\\
-1, & i=j,\\
0,  & \text{其他}.
\end{cases}
$$

边特征  

$$
y_{ij}=\mathrm{ReLU}\bigl([W_4 m_{ij}+b_4;\; W_5 a_{ij}+b_5]\bigr).
$$

#### GCN 编码器

**初始投影**  

$$
h_i^0 = W_{E1} x_i + b_{E1},\quad
h_{e_{ij}}^0 = W_{E2} y_{ij} + b_{E2}.
$$

随后经过 $L$ 层图卷积层，每层含**聚合**与**组合**两步。 
对节点 $i$ 的第 $\ell$ 层：

**聚合**  

$$
h_{N(i)}^\ell = \sigma\Bigl(W_I^\ell\cdot
\mathrm{ATTN}\bigl(h_i^{\ell-1},\lbrace h_u^{\ell-1}:u\in N(i)\rbrace \bigr)\Bigr),
$$

**组合**（带残差与层归一化）  

$$
h_i^\ell = \Bigl[V_I^\ell h_i^{\ell-1};\; h_{N(i)}^\ell\Bigr].
$$

对边 $e_{ij}$ 的第 $\ell$ 层：

**聚合**  

$$
h_{N(e_{ij})}^\ell = \sigma\Bigl(W_E^\ell\cdot
\bigl[W_{e1}^\ell h_{e_{ij}}^{\ell-1}+W_{e2}^\ell h_i^{\ell-1}+W_{e3}^\ell h_j^{\ell-1}\bigr]\Bigr),
$$

**组合**  

$$
h_{e_{ij}}^\ell = \Bigl[V_E^\ell h_{e_{ij}}^{\ell-1};\; h_{N(e_{ij})}^\ell\Bigr].
$$

#### 解码器

**序列预测解码器**

采用 GRU + Pointer 机制，逐步生成序列 $\pi$。 
在第 $t$ 步，GRU 隐状态 $z_t$ 汇总已生成部分 $\pi_{1:t-1}$，上下文权重  

$$
u_{ti}=
\begin{cases}
-\infty, & i\in N_{mt}\;(\text{不可行掩码}),\\[4pt]
v^\top \tanh\bigl(W^G [h_i^L;\, z_t]\bigr), & \text{否则}.
\end{cases}
$$

指向分布  

$$
p(\pi_t\mid \pi_{<t},S)=\mathrm{softmax}_i(u_{ti}).
$$

**容量掩码规则**  

- 已访问或需求 $x\_{d\_i}>\tilde c\_t$ 的节点被掩；  
- 仓库 0 不能在 $t=1$ 或连续两步被访问。

**边分类解码器**

将序列 $\pi$ 转成 0-1 边标签矩阵 $P_E^{\mathrm{VRP}^*}$（出现过的边为 1）。  
用 MLP 对每条边嵌入打分  

$$
p_{e_{ij}}^{\mathrm{VRP}}=\mathrm{softmax}\Bigl(\mathrm{MLP}(h_{e_{ij}}^L)\Bigr)\in[0,1]^2,
$$

以交叉熵逼近 $P_E^{\mathrm{VRP}^*}$，实现自监督。

#### 联合训练策略

**整体损失**  

$$
\mathcal{L}_\theta = \alpha\mathcal{L}_s(\theta)+\beta\mathcal{L}_r(\theta),
$$

其中  
- $\mathcal{L}_r$：REINFORCE 损失，baseline 为贪婪 rollout 成本 $b(S)$；  
- $\mathcal{L}_s$：边分类交叉熵损失，伪标签来自当前策略采样序列。

算法流程  
1. 每 epoch 用 rollout 策略更新 baseline；  
2. 采样 batch 实例 $\rightarrow$ GCN 编码 $\rightarrow$ 序列解码得 $\pi_i$；  
3. 将 $\pi_i$ 转为 0-1 边标签，训练边分类器；  
4. 联合反向传播更新参数 $\theta$。

## 实验

实验在实际数据集上进行。

![image-20260119161541832]({{site.url}}/img/2026-01-19-Efficiently_Solving_the_Practical_Vehicle_Routing_Problem_A_Novel_Joint_Learning_Approach/image-20260119161541832.png)





