---
layout:     post
title:      Rl4co an extensive reinforcement learning for combinatorial optimization benchmark
subtitle:   ICLR2024 通用、全面的RL4CO库
date:       2025/04/28
author:     Birdie
header-img: img/post\_header.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - ICLR
---

Rl4co: an extensive reinforcement learning for combinatorial optimization benchmark

一个用RL解决CO问题的算法库

![image-20250428161839667]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428161839667.png)

开源：[ai4co/rl4co: A PyTorch library for all things Reinforcement Learning (RL) for Combinatorial Optimization (CO)](https://github.com/ai4co/rl4co)

ICLR2024



## 贡献

第一个全面的baseline benchmark。

- 模块化27个环境和23个现有的baseline；
- 通过TorchRL、PyTorch Lightning、Hydra和TensorDict为NCO社区提供高效的训练和测试效率；
- 规范、公平、全面。能够自动测试不同分布的更广泛的问题。

## 方法分类

![image-20250428162321416]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428162321416.png)

## 结构

![image-20250428162420557]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428162420557.png)

### Environments

当前状态、当前解、策略选择的动作、奖励等静态信息和动态信息可以存储在字典中。

状态转移可以通过 reset 和 step 更新。

还可以检查解合法性和选择初始节点等不同的API。

### 策略

特征嵌入可以包含：节点嵌入、边嵌入、上下文嵌入等。

![image-20250428163259143]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428163259143.png)

### RL 方法

采用了多个日志记录系统如wandb。

该模块无缝支持现代训练管道的功能，包括日志记录，检查点管理，混合精度训练，各种硬件加速支持（例如，CPU，GPU，TPU和Apple Silicon）和分布式设置中的多设备硬件加速器。

甚至配备了FlashAttention。

### 工具

- 配置管理
- 解码方式：Greedy、Sampling（softmax、top-k、top-p）、Multistart、Augmentation
- 文档、教程、测试

![image-20250428163248637]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428163248637.png)

### 环境和baseline

环境

- 路由问题：TSP、CVRP、PCTSP、PDP、MTVRP（VRPTW、OVRP、VRPB、VRPL、其他VRPs）
- 调度：FJSSP、JSSP、FJSP
- 自动电子设计：mDPP（多开盖布局问题）
- 图：FLP（选址）、MCP（最大覆盖）

baseline

- 自回归构造：AM、Ptr-Net、POMO、MatNet、HAM、SymNCO、PolyNet、MTPOMO、MVMoE、L2D、HGNN、DevFormer
- 非自回归构造：DeepACO、GFACS、GLOP
- 改进方法：DAFCT、N2S、NeuOpt
- 通用RL：REINFORCE、A2C、PPO
- 后搜索：AS、EAS

## Benchmarking Study

### 模块化和灵活性

![image-20250428164640680]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428164640680.png)

可以换编码器和解码器，很灵活。

### 构造策略

快速公平地获得baseline。

![image-20250428164757498]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428164757498.png)

包括：解码的策略、泛化性、大规模实例。

### 结合构造和改进

![image-20250428165009659]({{site.url}}/img/2025-4-28-Rl4co_an_extensive_reinforcement_learning_for_combinatorial_optimization_benchmark/image-20250428165009659.png)













