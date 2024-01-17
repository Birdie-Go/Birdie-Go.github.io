---
layout:     post
title:      batch manufacturing
subtitle:   NIPS23
date:       2024/1/15
author:     Birdie
header-img: img/post_header.jpg
catalog: true
tags:
    - 随机过程
    - 强化学习
---

## 实验目的

Write programs to solve batch manufacturing problem

- Formulate and solve both discounted and average cost problems
- Realize both value iteration and policy iteration algorithms
- parameters $c, K, n, p, \alpha$ as inputs

Experiments and analysis

- Run the code with different parameter inputs to show the performance of value iteration and policy iteration
- For discounted problem, consider exercise 7.8
- For average cost problem, write code to find the threshold by stationary analysis
- Compare the optimal policies of the discounted problem wit hdifferent values of $\alpha$ and the average cost problem

## 前置知识

### Batch Manufacturing

A Manufacturer at each time period receives an order for her product with probability p and receives no order with probability $1 - p$.

At any period, she has a choice of processing all the unfilled orders in a batch, or process no order at all. The maximum number of orders that can remain unfilled is $n$.

The cost per unfilled order at each time period is $c > 0$, the setup
cost to process the unfilled orders is $K > 0$.

The manufacturer wants to find a processing policy that minimizes
the total expected cost with discount factor $\alpha < 1$.

### Discounted Problem

Discounted problems $(0<\alpha<1)$

$$
\lim _{N \rightarrow \infty} \underset{\substack{w_k \\ k=0,1, \ldots}}{E}\left\{\sum_{k=0}^{N-1} \alpha^k g\left(x_k, \mu_k\left(x_k\right), w_k\right)\right\}
$$

- $\alpha$ 表示衰减率
- $g(x_k, \mu_k\left(x_k\right), w_k)$ 表示从状态 $x_k$ 执行动作 $\mu_k(x_k)$ 到达状态$w_k$的代价

#### Value Iteration

给定初始状态 $J_0(1), \ldots, J_0(n)$, 使用动态规划方程计算 $J_k(i)$ 为

$$
J_{k+1}(i)=\min _{u \in U(i)}\left[g(i, u)+\alpha \sum_{j=1}^n p_{i j}(u) J_k(j)\right], \forall i
$$

最终所有状态会收敛到$J^\ast(i)$。

#### Policy Iteration

根据bellman方程，状态$i$的最优值函数应满足：

$$
J^\ast(i)=\min _{u \in U(i)}\left[g(i, u)+\alpha \sum_{j=1}^n p_{i j}(u) J^\ast(j)\right], \forall i
$$

对于一个给定的策略 $\mu$, 一个特定解的代价 $J_\mu(1), \cdots, J_\mu(n)$ 满足

$$
J_\mu(i)=g(i, \mu(i))+\alpha \sum_{j=1}^n p_{i j}(\mu(i)) J_\mu(j), \forall i
$$

给定初始状态 $J_0(1), \ldots, J_0(n)$, 使用动态规划方程计算 $J_k(i)$ 为

$$
J_{k+1}(i)=g(i, \mu(i))+\alpha \sum_{j=1}^n p_{i j}(\mu(i)) J_k(j), \forall i
$$

最终所有状态会收敛到 $J_{\mu}(i)$。

当且仅当对于每个状态 $i$，$\mu(i)$ 都达到Bellman方程中的最小值时，静态政策 $\mu$ 才是最优的。

因此策略迭代方程为

$$
\mu^{k+1}(i)=\arg \min _{u \in U(i)}\left[g(i, u)+\alpha \sum_{j=1}^n p_{i j}(u) J_{\mu^k}(j)\right], \forall i
$$

生成不断改进的策略序列，并以最优策略结束。

### Average Cost Problem

Average cost per stage problems

$$
\lim _{N \rightarrow \infty} \frac{1}{N} \underset{\substack{w_k \\ k=0,1, \ldots}}{E}\left\{\sum_{k=0}^{N-1} g\left(x_k, \mu_k\left(x_k\right), w_k\right)\right\}
$$

- $g(x_k, \mu_k\left(x_k\right), w_k)$表示从状态 $x_k$ 执行动作 $\mu_k(x_k)$ 到达状态$w_k$的代价

#### Value Iteration

Bellman方程如下：

$$
\lambda^\ast+h^\ast(i)=\min _{u \in U(i)}\left[g(i, u)+\sum_{j=1}^n p_{i j}(u) h^\ast(j)\right], \forall i
$$


未知变量 $\lambda^\ast, h^\ast(i)$，$\lambda^\ast$ 表示最优平均cost，$h^\ast$ 表示各个状态的最优cost与平均最优cost的差分cost。

给定 $h_k(i)$，以及固定状态 $s$，计算

$$
\lambda_k=\min _{u \in U(s)}\left[g(s, u)+\sum_{j=1}^n p_{s j}(u) h_k(j)\right]
$$

和

$$
h_{k+1}(i)=\min _{u \in U(i)}\left[g(i, u)+\sum_{j=1}^n p_{i j}(u) h_k(j)\right]-\lambda_k
$$

#### Policy Iteration

在第$k$轮，固定策略 $\mu^k$，计算价值函数

$$
\begin{gathered}
\lambda^k+h^k(i)=g\left(i, \mu^k(i)\right)+\sum_{j=1}^n p_{i j}\left(\mu^k(i)\right) h^k(j), \forall i \\
h^k(n)=0
\end{gathered}
$$

策略改进：

$$
\mu^{k+1}(i)=\arg \min _{u \in U(i)}\left[g(i, u)+\sum_{j=1}^n p_{i j}(u) h^k(j)\right], \forall i
$$

终止状态为 $\lambda^{k+1}=\lambda^k, h^{k+1}(i)=h^k(i), \forall i$。

\section{方法}

### 马尔科夫决策建模

- State $i \in\{0,1, \cdots, n\}$ : number of unfilled orders
- Action $u \in\{0,1\}$ : process (1) or not (0)

  $$
  u \in\{0,1\}, \text { if } i<n ; \quad u=1 \text {, if } i=n
  $$

- State Transition $p_{i j}(u)$ :

  $$
  \begin{gathered}
  p_{i 1}(1)=p_{i(i+1)}(0)=p, \quad p_{i 0}(1)=p_{i i}(0)=1-p, \quad i<n \\
  p_{n 1}(1)=p, \quad p_{n 0}(1)=1-p
  \end{gathered}
  $$

- Per-stage cost

  $$
  g(i, 1)=K, \quad g(i, 0)=c i
  $$

### 在Discounted Problem下

#### Value Iteration

初始值

$$
J_0(i) = 0,\forall i
$$

Value iteration

$$
\begin{aligned}
J_{k+1}(i)= & \min \left[K+\alpha(1-p) J_k(0)+\alpha p J_k(1)\right. \\
& \left.c i+\alpha(1-p) J_k(i)+\alpha p J_k(i+1)\right], \quad i=0,1, \cdots, n-1, \\
J_{k+1}(n)= & K+\alpha(1-p) J_k(0)+\alpha p J_k(1)
\end{aligned}
$$

Convergence:

$$
J^\ast(i)=\lim _{k \rightarrow \infty} J_k(i)
$$

#### Policy Iteration

初始值

$$
\mu^0(i)=1, \forall i
$$

策略评估

$$
\begin{aligned}
J_{\mu^{k+1}}(i)&=\left\{
	\begin{aligned}
	K+\alpha(1-p) J_{\mu^k}(0)+\alpha p J_{\mu^k}(1) \quad \mu^k(i)=0\\
	c i+\alpha(1-p) J_{\mu^k}(i)+\alpha p J_{\mu^k}(i+1) \quad \mu^k(i)=1\\
	\end{aligned}
	\right
	.
    \quad\text{if} \quad i=0,1, \cdots, n-1,\\
    J_{\mu^{k+1}}(n)&= K+\alpha(1-p) J_{\mu^k}(0)+\alpha p J_{\mu^k}(1)
\end{aligned}
$$

策略改进

$$
\begin{aligned}
    \text{cost}_{process}(i) &= K+(1-p) J_{\mu^k}(0)+ p J_{\mu^k}(1)+\lambda_{k}\\
    \text{cost}_{unprocess}(i) &= c i+(1-p) J_{\mu^k}(i)+ p J_{\mu^k}(i+1)+\lambda_{k}\\
    \mu^{k+1}(i)&=\left\{
        \begin{aligned}
            0 \quad \text{cost}_{process}(i)\geq\text{cost}_{unprocess}(i)\\
            1 \quad \text{cost}_{process}(i)<\text{cost}_{unprocess}(i)\\
        \end{aligned}
    \right. \quad \text{ if } i = 0, \cdots, n-1\\
    \mu^{k+1}(n)&=1
\end{aligned}
$$

### 在Average Cost Problem下

Bellman方程为：

$$
\begin{aligned}
    \lambda^\ast+h^\ast(i)= & \min \left[K+(1-p) h^\ast(0)+p h^\ast(1)\right. \\
    & \left.c i+(1-p) h^\ast(i)+p h^\ast(i+1)\right], \quad i=0,1, \cdots, n-1 \\
    \lambda^\ast+h^\ast(n)= & K+(1-p) h^\ast(0)+p h^\ast(1)
\end{aligned}
$$

#### Value Iteration

初始值

$$
J_0(i) = 0,\forall i
$$

固定状态为$0$，每一轮循环必定会从0开始并返回0。

Value iteration

$$
\begin{aligned}
\lambda_{k}=&(1 - p) * J_k(0) + p * J_k(1)\\
J_{k+1}(i)= & \min \left[K+(1-p) J_k(0)+ p J_k(1)\right. \\
& \left.c i+(1-p) J_k(i)+ p J_k(i+1)\right]-\lambda_{k}, \quad i=0,1, \cdots, n-1, \\
J_{k+1}(n)= & K+(1-p) J_k(0)+ p J_k(1)-\lambda_{k}
\end{aligned}
$$

Convergence:

$$
J^\ast(i)=\lim _{k \rightarrow \infty} J_k(i),\lambda^\ast=\lim _{k \rightarrow \infty} \lambda_k
$$

#### Policy Iteration

初始值

$$
\mu^0(i)=1, \forall i
$$

策略评估

$$
\begin{aligned}
\lambda_{k}&=(1 - p) * J_k(0) + p * J_k(1)\\
J_{\mu^{k+1}}(i)&=\left\{
	\begin{aligned}
	K+(1-p) J_{\mu^k}(0)+ p J_{\mu^k}(1)-\lambda_{k} \quad \mu^k(i)=0\\
	c i+(1-p) J_{\mu^k}(i)+ p J_{\mu^k}(i+1)-\lambda_{k} \quad \mu^k(i)=1\\
	\end{aligned}
	\right
	.
    \quad\text{if} \quad i=0,1, \cdots, n-1,\\
J_{\mu^{k+1}}(n)&= K+(1-p) J_{\mu^k}(0)+ p J_{\mu^k}(1)-\lambda_{k}
\end{aligned}
$$

策略改进

$$
\begin{aligned}
    \text{cost}_{process}(i) &= K+(1-p) J_{\mu^k}(0)+ p J_{\mu^k}(1)+\lambda_{k}\\
    \text{cost}_{unprocess}(i) &= c i+(1-p) J_{\mu^k}(i)+ p J_{\mu^k}(i+1)+\lambda_{k}\\
    \mu^{k+1}(i)&=\left\{
        \begin{aligned}
            0 \quad \text{cost}_{process}(i)\geq\text{cost}_{unprocess}(i)\\
            1 \quad \text{cost}_{process}(i)<\text{cost}_{unprocess}(i)\\
        \end{aligned}
    \right. \quad \text{ if } i = 0, \cdots, n-1\\
    \mu^{k+1}(n)&=1
\end{aligned}
$$

### 根据value function得到policy

正常来说，我们可以通过value-action function采用贪婪的策略得到policy，但在本次实验中，通过value iteration会得到状态价值函数，状态价值函数只能评估一个状态的好坏程度。已知问题是batch manufacturing problem，决策只有process和unprocess两种，从贪婪的角度来说，如果unporcess能够得到更好的状态，那么我们选择unporcess，否则将会process。因此，在得到value function后，所有拥有最大value的状态，都会得到process的决策，其他状态则为unprocess。

## 实现

### Discounted Problem下的Value Iteration

```python
def _iteration_discounted(self, p, n, c, K, alpha):
    # 当前的Value函数表，初始值为零向量
    value = [0] * (n + 1)
    # 上一轮的Value函数表，初始值为零向量
    last_value = [0] * (n + 1)
    # 记录迭代轮数
    turns = 0
    while True:
        # 轮次 + 1
        turns = turns + 1
        # 根据更新方程，更新value[0...n-1]
        for i in range(n):
            value[i] = min(
                K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1],
                c * i + alpha * (1 - p) * last_value[i] + alpha * p * last_value[i + 1]
            )
        # 根据更新方程，更新value[n]
        value[n] = K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1]
        # 判断是否收敛，即两个向量的差小于EPS
        if is_terminal(value, last_value, Value.EPS):
            break
        last_value = value.copy()
    
    return {
        "Iteration turns" : turns,
        "Value function" : format_float(value),
        "Policy" : self._get_policy(value)
    }
```

### Discounted Problem下的Policy Iteration

```python
# 策略评估：根据当前策略policy计算value表
def _get_value_discounted(self, p, n, c, K, alpha, policy):
    value = [0] * (n + 1)
    last_value = [0] * (n + 1)
    turns = 0
    while True:
        turns = turns + 1
        for i in range(n):
            value[i] = (
                K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1] if policy[i] else
                c * i + alpha * (1 - p) * last_value[i] + alpha * p * last_value[i + 1]
            )
        value[n] = K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1]
        if is_terminal(value, last_value, Policy.EPS):
            break
        last_value = value.copy()
    
    return value, turns

# 策略改进：策略迭代
def _iteration_discounted(self, p, n, c, K, alpha):
    # 当前轮的策略，目前是初始策略
    policy = [0] * n + [1]
    # 上一轮的策略
    last_policy = [0] * n + [1]
    # 记录轮数
    turns = 0
    # 价值计算总轮数
    value_turns = 0
    while True:
        # 轮数 + 1
        turns = turns + 1
        # 根据上一轮的策略计算value表
        value, turns_i = self._get_value_discounted(p, n, c, K, alpha, last_policy)
        value_turns += turns_i

        # 根据value表计算最新的策略
        for i in range(n):
            # 如果process，代价为
            process_cost = K + alpha * (1 - p) * value[0] + alpha * p * value[1]
            # 如果不process，代价为
            unprocess_cost = c * i + alpha * (1 - p) * value[i] + alpha * p * value[i + 1]
            # 选取最小值对应的动作
            policy[i] = 1 if process_cost < unprocess_cost else 0
        # policy[n]的动作空间只有{1}
        policy[n] = 1

        # 判断是否达到终止状态，即value表收敛
        if policy == last_policy:
            break
        last_policy = policy.copy()

    return {
        "Policy improvement iteration turns" : turns,
        "Calucating value function total iteration turns" : value_turns,
        "Value function" : format_float(value),
        "Policy" : policy
    }
```

### Average Cost Problem下的Value Iteration

```python
def _iteration_average(self, p, n, c, K):
    # 当前轮次的差分价值函数
    value_h = [0] * (n + 1)
    # 上一轮次的差分价值函数
    last_value_h = [0] * (n + 1)
    # 上一轮次的平均价值函数
    last_lambda_k = 0
    # 记录轮数
    turns = 0
    while True:
        # 轮数 + 1
        turns = turns + 1
        # 固定状态0，计算平均价值函数
        lambda_k = min(
            K + (1 - p) * last_value_h[0] + p * last_value_h[1],
            (1 - p) * last_value_h[0] + p * last_value_h[1]
        )
        # 迭代计算当前轮次差分状态函数
        for i in range(n):
            value_h[i] = min(
                K + (1 - p) * last_value_h[0] + p * last_value_h[1],
                c * i + (1 - p) * last_value_h[i] + p * last_value_h[i + 1]
            ) - lambda_k
        # 计算状态n的当前轮次差分状态函数
        value_h[n] = K + (1 - p) * last_value_h[0] + p * last_value_h[1] - lambda_k
        # 判断是否达到终止状态
        if is_terminal(value_h, last_value_h, Value.EPS) and abs(last_lambda_k - lambda_k) < Value.EPS:
            break
        # 更新轮次状态
        last_value_h = value_h.copy()
        last_lambda_k = lambda_k
    
    value = [i + lambda_k for i in value_h]
    
    return {
        "Iteration turns" : turns,
        "Value function" : format_float(value),
        "Policy" : self._get_policy(value),
        "Optimal average cost" : format_float(lambda_k)
    }
```

### Average Cost Problem下的Policy Iteration

```python
# 策略评估：根据当前轮次的策略，计算当前轮次的平均价值和差分价值表
def _get_value_average(self, p, n, c, K, policy):
    value_h = [0] * (n + 1)
    last_value_h = [0] * (n + 1)
    last_lambda_k = 0
    turns = 0
    while True:
        turns = turns + 1
        lambda_k = min(
            K + (1 - p) * last_value_h[0] + p * last_value_h[1],
            (1 - p) * last_value_h[0] + p * last_value_h[1]
        )
        for i in range(n):
            value_h[i] = (
                K + (1 - p) * last_value_h[0] + p * last_value_h[1] if policy[i] else
                c * i + (1 - p) * last_value_h[i] + p * last_value_h[i + 1]
            ) - lambda_k
        value_h[n] = K + (1 - p) * last_value_h[0] + p * last_value_h[1] - lambda_k
        if is_terminal(value_h, last_value_h, Policy.EPS) and abs(last_lambda_k - lambda_k) < Policy.EPS:
            break
        last_value_h = value_h.copy()
        last_lambda_k = lambda_k
    
    return lambda_k, value_h, turns

# 策略改进：策略迭代
def _iteration_average(self, p, n, c, K):
    # 初始化当前轮次的策略
    policy = [1] * (n + 1)
    # 初始化上一轮次的策略
    last_policy = [1] * (n + 1)
    # 记录轮次
    turns = 0
    # 价值函数迭代轮数
    value_turns = 0
    while True:
        # 轮次 + 1
        turns = turns + 1
        # 根据当前轮次的策略，计算当前轮次的平均价值和差分价值表
        value, turns_i = self._get_value_average(p, n, c, K, last_policy)
        value_turns += turns_i

        # 策略迭代
        for i in range(n):
            process_cost = K + (1 - p) * value[0] + p * value[1]
            unprocess_cost = c * i + (1 - p) * value[i] + p * value[i + 1]
            policy[i] = 1 if process_cost < unprocess_cost else 0
        policy[n] = 1

        # 判断是否达到终止状态
        if policy == last_policy:
            break
        last_policy = policy.copy()

    return {
        "Policy improvement iteration turns" : turns,
        "Calucating value function total iteration turns" : value_turns,
        "Value function" : format_float(value),
        "Policy" : policy
    }
```

### 工具函数

```python
# 用与判断两个list：x和y每个元素的距离都小于EPS
def is_terminal(x, y, EPS):
    if x is None or y is None:
        return False

    for (xi, yi) in zip(x, y):
        if abs(xi - yi) > EPS:
            return False
    return True

# 用语规范list或者float为4位小数
def format_float(v):
    if type(v) == type(list()):
        return [float('{:.4f}'.format(i)) for i in v]
    else:
        return round(v, 4)

# 根据value function得到policy
def _get_policy(self, value):
    max_value = max(value)
    policy = [0 if i < max_value - Value.EPS else 1 for i in value]
    return policy
```
    
## 实验

实验指令见附件。

### Discounted Problem

#### 比较不同的 $n$

```python
Used value iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 8, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 12, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 8, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1]

p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

p = 0.5, n = 12, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

一方面，只要其他参数不改变，当process的阈值小于n，value function和policy并不会随着$n$的增长而改变；另一方面，在这组参数下，value iteration和policy iteration得到的value function和policy是一致的，尽管policy iteration所需要的策略改进轮数更少，但其进行策略评估的总轮数是要大于value iteration所需要的轮数的。

#### 比较不同的 $K$

```python
Used value iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 1.0, K = 3.0, alpha = 0.9.
    Iteration turns : 91
    Value function : [10.5741, 12.9241, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 8.0, alpha = 0.9.
    Iteration turns : 98
    Value function : [18.358, 22.4377, 25.2018, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 1.0, K = 3.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 378
    Value function : [10.5741, 12.9241, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741, 13.5741]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 8.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 407
    Value function : [18.358, 22.4377, 25.2018, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358, 26.358]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```

当 $K$ 发生变化时，两种方法得到的结果仍然是一致的；尽管policy iteration所需要的策略改进轮数更少，但其进行策略评估的总轮数是要大于value iteration所需要的轮数的。

当 $K$ 增大时，所有状态的价值函数的值都会增大，而得到的最优策略显示，货物会存放更多的天数，这样也是符合常识的。

#### 比较不同的 $c$

```python
Used value iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 0.5, K = 5.0, alpha = 0.9.
    Iteration turns : 92
    Value function : [10.319, 12.6123, 14.3042, 15.2609, 15.319, 15.319, 15.319, 15.319, 15.319, 15.319, 15.319]
    Policy : [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 2.0, K = 5.0, alpha = 0.9.
    Iteration turns : 97
    Value function : [19.1242, 23.3742, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 0.5, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 387
    Value function : [10.3191, 12.6124, 14.3042, 15.2609, 15.3191, 15.3191, 15.3191, 15.3191, 15.3191, 15.3191, 15.3191]
    Policy : [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 2.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 398
    Value function : [19.1242, 23.3742, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242, 24.1242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

当 $c$ 发生变化时，两种方法得到的结果仍然是一致的；尽管policy iteration所需要的策略改进轮数更少，但其进行策略评估的总轮数是要大于value iteration所需要的轮数的。

当 $c$ 增大时，所有状态的价值函数的值都会增大，而得到的最优策略显示，货物会存放更少的天数，这样也是符合常识的。

#### 比较不同的$\alpha$

```python
Used value iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.0.
    Iteration turns : 2
    Value function : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    Policy : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.1.
    Iteration turns : 6
    Value function : [0.0617, 1.1728, 2.2839, 3.3935, 4.4769, 5.0617, 5.0617, 5.0617, 5.0617, 5.0617, 5.0617]
    Policy : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.5.
    Iteration turns : 16
    Value function : [0.9615, 2.8845, 4.6538, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.0.
    Policy improvement iteration turns : 2
    Calucating value function total iteration turns : 4
    Value function : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    Policy : [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.1.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 18
    Value function : [0.0617, 1.1728, 2.2839, 3.3935, 4.4769, 5.0617, 5.0617, 5.0617, 5.0617, 5.0617, 5.0617]
    Policy : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.5.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 49
    Value function : [0.9615, 2.8846, 4.6538, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615, 5.9615]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

上述实验可以反应，在discounted problem下，value iteration和policy iteration得到的结果是一致的，只有迭代轮数上的差异。

$\alpha$ 越接近1，代表对未来奖励的重视程度越高；而 $\alpha$ 越接近0，代表对未来奖励的重视程度越低。当$\alpha$增大时，

- 上一轮迭代的结果对当前轮数的影响会增加，收敛的速度会减缓，迭代的轮数会增加；
- 上一轮迭代的结果会以一定程度的衰减增加到当前的value function上，因此最终收敛后的value function会增大；
- policy会更加注重长期收益，而随着轮数的增加，process的收益是最大的，因此policy中process的决策点会提前；
- 当$\alpha=1$时，奖励会一直叠加导致iteration无法终止。

#### 比较不同的 $p$

```python
Used value iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.1, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 85
    Value function : [4.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991]
    Policy : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 95
    Value function : [14.6242, 17.8742, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242, 19.6242]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.9, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 98
    Value function : [21.1872, 23.803, 25.5072, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with discounted problem:
p = 0.1, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 284
    Value function : [4.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991, 9.4991]
    Policy : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 297
    Value function : [14.6241, 17.8741, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241, 19.6241]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
p = 0.9, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 401
    Value function : [21.1872, 23.8031, 25.5072, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872, 26.1872]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```

上述实验可以反应，在discounted problem下，value iteration和policy iteration得到的结果是一致的，只有迭代轮数上的差异。

当$p$增大时，获得订单的概率会增大，

- 因此每一轮unprocess的收益会增大，下一个时刻很可能又获得一个订单使得下一个时刻process的收益增加，因此process的决策点会推移；
- 由于决策点会往后推移，因此计算的value function会整体增大；
- 计算的value function会整体增大，收敛的轮数也会增加。

### Average cost Problem

#### 比较不同的 $n$

```python
Used value iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 8, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 12, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 8, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 5
    Calucating value function total iteration turns : 100
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 5
    Calucating value function total iteration turns : 100
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 12, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 5
    Calucating value function total iteration turns : 100
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
```

可以观察，

- 当 $n$ 本身已经超过unprocess决策窗口时，随着$n$的增大，value iteration和policy iteration的结果并不会变化，包括迭代的次数也不会变化；
- 得到的value function可以应证，状态价值函数是一个递增的序列，也是符合预期的，同时value[0]=optimal average cost也是符合预期的；
- 横向比较看，value iteration和policy iteration得到的状态价值函数和策略都是一样的；
- 从迭代轮数来看，policy iteration所需要的计算代价会更大。

#### 比较不同的 $K$

```python
Used value iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 10, c = 1.0, K = 3.0, alpha = 0.9.
    Iteration turns : 5
    Value function : [1.25, 3.75, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.25
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 10, c = 1.0, K = 8.0, alpha = 0.9.
    Iteration turns : 18
    Value function : [2.3333, 7.0001, 9.6667, 10.3333, 10.3333, 10.3333, 10.3333, 10.3333, 10.3333, 10.3333, 10.3333]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.3333
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 10, c = 1.0, K = 3.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 42
    Value function : [1.25, 3.75, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25, 4.25]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.25
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 5
    Calucating value function total iteration turns : 100
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 10, c = 1.0, K = 8.0, alpha = 0.9.
    Policy improvement iteration turns : 6
    Calucating value function total iteration turns : 251
    Value function : [2.3334, 7.0001, 9.6667, 10.3334, 10.3334, 10.3334, 10.3334, 10.3334, 10.3334, 10.3334, 10.3334]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.3334
```

当 $K$ 发生变化时，两种方法得到的结果仍然是一致的；尽管policy iteration所需要的策略改进轮数更少，但其进行策略评估的总轮数是要大于value iteration所需要的轮数的。

当 $K$ 增大时，所有状态的价值函数的值都会增大，而得到的最优策略显示，货物会存放更多的天数，这样也是符合常识的。

可以观察，

- 两种方法得到的结果都是一致的；
- 当 $K$ 增大时，所有状态的价值函数的值都会增大，而得到的最优策略显示，货物会存放更多的天数，这样也是符合常识的。
- 当 $K$ 增大时，所需要的计算代价会更大，收敛的时间会更久。

#### 比较不同的$c$

```python
Used value iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 10, c = 0.5, K = 5.0, alpha = 0.9.
    Iteration turns : 18
    Value function : [1.3333, 4.0, 5.6667, 6.3333, 6.3333, 6.3333, 6.3333, 6.3333, 6.3333, 6.3333, 6.3333]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.3333
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.5, n = 10, c = 2.0, K = 5.0, alpha = 0.9.
    Iteration turns : 5
    Value function : [2.25, 6.75, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.25
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.5, n = 10, c = 0.5, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 240
    Value function : [1.375, 4.1251, 5.875, 6.625, 6.375, 6.375, 6.375, 6.375, 6.375, 6.375, 6.375]
    Policy : [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.375
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 96
    Value function : [1.8334, 5.5001, 7.1667, 6.8334, 6.8334, 6.8334, 6.8334, 6.8334, 6.8334, 6.8334, 6.8334]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.8334
    
p = 0.5, n = 10, c = 2.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 28
    Value function : [2.25, 6.75, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25, 7.25]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.25
```

当 $c$ 增大时，所有状态的价值函数的值都会增大，而得到的最优策略显示，货物会存放更少的天数，这样也是符合常识的。

#### 比较不同的 $p$

```python
Used value iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.1, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 8
    Value function : [0.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
    Policy : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 0.5
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 6
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
p = 0.9, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Iteration turns : 62
    Value function : [2.5, 5.2777, 6.9444, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.5
```

```python
Used policy iteration to solve batch manufacturing problem, formulate with average_cost problem:
p = 0.1, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 3
    Calucating value function total iteration turns : 231
    Value function : [0.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
    Policy : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 0.5
    
    
p = 0.5, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 5
    Calucating value function total iteration turns : 100
    Value function : [1.75, 5.25, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75]
    Policy : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 1.75
    
    
p = 0.9, n = 10, c = 1.0, K = 5.0, alpha = 0.9.
    Policy improvement iteration turns : 4
    Calucating value function total iteration turns : 338
    Value function : [2.5, 5.2778, 6.9444, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
    Policy : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    Optimal average cost : 2.5
```

上述实验可以反应，在average cost problem下，value iteration和policy iteration得到的结果是一致的，但迭代轮数上的也有差异。

当 $p$ 增大时，获得订单的概率会增大，

- 因此每一轮unprocess的收益会增大，下一个时刻很可能又获得一个订单使得下一个时刻process的收益增加，因此process的决策点会推移；
- 由于决策点会往后推移，因此计算的value function会整体增大；
- 计算的value function会整体增大，收敛的轮数也会增加。

### 实验总结

具体的计算结果、比较和分析见上述实验。


- 在Discounted problem下，Value iteration和Policy iteration得到的结果是一致的，只有计算代价上的区别，后者的计算开销会更大，因为其不仅要策略改进，还需要进行策略的评估，而策略评估的开销是比较大的；
- 在Average cost problem下，Value iteration和Policy iteration得到的结果也是是一致的，同样的，也是后者的计算开销也会更大；
- 当 $n$ 增加时，只要threshold在 $n$ 的范围内，对结果是没有影响的；
- 当 $c$ 增加时，所有状态的价值函数的值都会增大，计算开销会减小，而得到的最优策略显示，货物会存放更少的天数，这样也是符合常识的；
- 当 $K$ 增加时，所有状态的价值函数的值都会增大，计算开销会增大，而得到的最优策略显示，货物会存放更多的天数，这样也是符合常识的；
- 当 $\alpha$ 增加时，上一轮迭代的结果对当前轮数的影响会增加，收敛的速度会减缓，迭代的轮数会增加，上一轮迭代的结果会以一定程度的衰减增加到当前的value function上，因此最终收敛后的value function会增大，policy会更加注重长期收益，而随着轮数的增加，process的收益是最大的，因此policy中process的决策点会提前；
- 当 $p$ 增大时，因此每一轮unprocess的收益会增大，下一个时刻很可能又获得一个订单使得下一个时刻process的收益增加，因此process的决策点会推移，由于决策点会往后推移，因此计算的value function会整体增大，计算的value function会整体增大，收敛的轮数也会增加。

## 附件

### 代码

`main.py`

```python
import argparse

from algorithm import Value, Policy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to solve batch manufacturing problem.')

    parser.add_argument('--p', type=float, nargs='+', default=[0.5], help='Probability of receiving an order for the product.')
    parser.add_argument('--n', type=int, nargs='+', default=[5], help='The maximum number of orders that can remain unfilled.')
    parser.add_argument('--c', type=float, nargs='+', default=[1.0], help='The cost per unfilled order at each time period.')
    parser.add_argument('--K', type=float, nargs='+', default=[3.0], help='The setup cost to process the unfilled orders.')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.8], help='Discount factor.')
    
    parser.add_argument('--problem', type=str, nargs='+', choices=['discounted', 'average_cost'], default=['discounted', 'average_cost'], help='Problem in [discounted, average_cost].')
    parser.add_argument('--algorithm', type=str, choices=['value', 'policy'], nargs='+', default=['value', 'policy'], help='Algorithm in [value, policy] iteration.')
    
    args = parser.parse_args()

    for algorithm in args.algorithm:

        for problem in args.problem:

            method = {
                'value' : Value(problem),
                'policy' : Policy(problem)
            }.get(algorithm)
            print('Used {} iteration to solve batch manufacturing problem, formulate with {} problem:'.format(algorithm, problem))

            for p in args.p:
                for n in args.n:
                    for c in args.c:
                        for K in args.K:
                            for alpha in args.alpha:
                                result = method.iteration(p, n, c, K, alpha)
                                print('p = {}, n = {}, c = {}, K = {}, alpha = {}.'.format(p, n, c, K, alpha))
                                for key, value in result.items():
                                    print(key, ":", value)
                                print("\n")
```

`utils.py`

```python
def is_terminal(x, y, EPS):
    
    if x is None or y is None:
        return False

    for (xi, yi) in zip(x, y):
        if abs(xi - yi) > EPS:
            return False
    return True

def format_float(v):
    if type(v) == type(list()):
        return [float('{:.4f}'.format(i)) for i in v]
    else:
        return round(v, 4)
```

`algorithm.py`

```python
from utils import is_terminal, format_float

class Value:

    EPS = 1e-4

    def __init__(self, problem):
        self.problem = problem

    def _get_policy(self, value):
        max_value = max(value)
        policy = [0 if i < max_value - Value.EPS else 1 for i in value]
        return policy
    
    def _iteration_discounted(self, p, n, c, K, alpha):
        value = [0] * (n + 1)
        last_value = [0] * (n + 1)
        turns = 0
        while True:
            turns = turns + 1
            for i in range(n):
                value[i] = min(
                    K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1],
                    c * i + alpha * (1 - p) * last_value[i] + alpha * p * last_value[i + 1]
                )
            value[n] = K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1]
            if is_terminal(value, last_value, Value.EPS):
                break
            last_value = value.copy()
        
        return {
            "Iteration turns" : turns,
            "Value function" : format_float(value),
            "Policy" : self._get_policy(value)
        }
    
    def _iteration_average(self, p, n, c, K):
        value_h = [0] * (n + 1)
        last_value_h = [0] * (n + 1)
        lambda_k = last_lambda_k = 0
        turns = 0
        while True:
            turns = turns + 1
            lambda_k = min(
                K + (1 - p) * last_value_h[0] + p * last_value_h[1],
                (1 - p) * last_value_h[0] + p * last_value_h[1]
            )
            for i in range(n):
                value_h[i] = min(
                    K + (1 - p) * last_value_h[0] + p * last_value_h[1],
                    c * i + (1 - p) * last_value_h[i] + p * last_value_h[i + 1]
                ) - lambda_k
            value_h[n] = K + (1 - p) * last_value_h[0] + p * last_value_h[1] - lambda_k
            if is_terminal(value_h, last_value_h, Value.EPS) and abs(last_lambda_k - lambda_k) < Value.EPS:
                break
            last_value_h = value_h.copy()
            last_lambda_k = lambda_k
        
        value = [i + lambda_k for i in value_h]
        
        return {
            "Iteration turns" : turns,
            "Value function" : format_float(value),
            "Policy" : self._get_policy(value),
            "Optimal average cost" : format_float(lambda_k)
        }


    def iteration(self, p, n, c, K, alpha):
        return {
            'discounted' : self._iteration_discounted(p, n, c, K, alpha),
            'average_cost' : self._iteration_average(p, n, c, K)
        }.get(self.problem)


class Policy:

    EPS = 1e-4

    def __init__(self, problem):
        self.problem = problem
    
    def _get_value_discounted(self, p, n, c, K, alpha, policy):
        value = [0] * (n + 1)
        last_value = [0] * (n + 1)
        turns = 0
        while True:
            turns = turns + 1
            for i in range(n):
                value[i] = (
                    K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1] if policy[i] else
                    c * i + alpha * (1 - p) * last_value[i] + alpha * p * last_value[i + 1]
                )
            value[n] = K + alpha * (1 - p) * last_value[0] + alpha * p * last_value[1]
            if is_terminal(value, last_value, Policy.EPS):
                break
            last_value = value.copy()
        
        return value, turns
    
    def _get_value_average(self, p, n, c, K, policy):
        value_h = [0] * (n + 1)
        last_value_h = [0] * (n + 1)
        last_lambda_k = 0
        turns = 0
        while True:
            turns = turns + 1
            lambda_k = min(
                K + (1 - p) * last_value_h[0] + p * last_value_h[1],
                (1 - p) * last_value_h[0] + p * last_value_h[1]
            )
            for i in range(n):
                value_h[i] = (
                    K + (1 - p) * last_value_h[0] + p * last_value_h[1] if policy[i] else
                    c * i + (1 - p) * last_value_h[i] + p * last_value_h[i + 1]
                ) - lambda_k
            value_h[n] = K + (1 - p) * last_value_h[0] + p * last_value_h[1] - lambda_k
            if is_terminal(value_h, last_value_h, Policy.EPS) and abs(last_lambda_k - lambda_k) < Policy.EPS:
                break
            last_value_h = value_h.copy()
            last_lambda_k = lambda_k
        
        return lambda_k, value_h, turns
    
    def _iteration_discounted(self, p, n, c, K, alpha):
        policy = [1] * (n + 1)
        last_policy = [1] * (n + 1)
        turns = 0
        value_turns = 0
        while True:
            turns = turns + 1
            value, turns_i = self._get_value_discounted(p, n, c, K, alpha, last_policy)
            value_turns += turns_i

            for i in range(n):
                process_cost = K + alpha * (1 - p) * value[0] + alpha * p * value[1]
                unprocess_cost = c * i + alpha * (1 - p) * value[i] + alpha * p * value[i + 1]
                policy[i] = 1 if process_cost < unprocess_cost else 0
            policy[n] = 1

            if policy == last_policy:
                break
            last_policy = policy.copy()

        return {
            "Policy improvement iteration turns" : turns,
            "Calucating value function total iteration turns" : value_turns,
            "Value function" : format_float(value),
            "Policy" : policy
        }
    
    def _iteration_average(self, p, n, c, K):
        policy = [1] * (n + 1)
        last_policy = [1] * (n + 1)
        turns = 0
        value_turns = 0
        while True:
            turns = turns + 1
            lambda_k, value, turns_i = self._get_value_average(p, n, c, K, last_policy)
            value_turns += turns_i

            for i in range(n):
                process_cost = K + (1 - p) * value[0] + p * value[1]
                unprocess_cost = c * i + (1 - p) * value[i] + p * value[i + 1]
                policy[i] = 1 if process_cost < unprocess_cost else 0
            policy[n] = 1

            if last_policy == policy:
                break
            last_policy = policy.copy()
        
        value_f = [i + lambda_k for i in value]

        return {
            "Policy improvement iteration turns" : turns,
            "Calucating value function total iteration turns" : value_turns,
            "Value function" : format_float(value_f),
            "Policy" : policy,
            "Optimal average cost" : format_float(lambda_k)
        }

    def iteration(self, p, n, c, K, alpha):
        return {
            'discounted' : self._iteration_discounted(p, n, c, K, alpha),
            'average_cost' : self._iteration_average(p, n, c, K)
        }.get(self.problem)
```

### 指令

### discounted problem实验

#### 指令1.1 比较不同的n

`python main.py --n 8 10 12 --K 5 --c 1 --alpha 0.9 --p 0.5 --problem discounted --algorithm value policy`

#### 指令1.2 比较不同的K

`python main.py --n 10 --K 3 5 8 --c 1 --alpha 0.9 --p 0.5 --problem discounted --algorithm value policy`

#### 指令1.3 比较不同的c

`python main.py --n 10 --K 5 --c 0.5 1 2 --alpha 0.9 --p 0.5 --problem discounted --algorithm value policy`

#### 指令1.4 比较不同的alpha

`python main.py --n 10 --K 5 --c 1 --alpha 0 0.1 0.5 0.9 --p 0.5 --problem discounted --algorithm value policy`

#### 指令1.5 比较不同的p

`python main.py --n 10 --K 5 --c 1 --alpha 0.9 --p 0.1 0.5 0.9 --problem discounted --algorithm value policy`

### average cost problem实验

#### 指令2.1 比较不同的n

`python main.py --n 8 10 12 --K 5 --c 1 --alpha 0.9 --p 0.5 --problem average_cost --algorithm value policy`

#### 指令2.2 比较不同的K

`python main.py --n 10 --K 3 5 8 --c 1 --alpha 0.9 --p 0.5 --problem average_cost --algorithm value policy`

#### 指令2.3 比较不同的c

`python main.py --n 10 --K 5 --c 0.5 1 2 --alpha 0.9 --p 0.5 --problem average_cost --algorithm value policy`

#### 指令2.4 比较不同的p

`python main.py --n 10 --K 5 --c 1 --alpha 0.9 --p 0.1 0.5 0.9 --problem average_cost --algorithm value policy`