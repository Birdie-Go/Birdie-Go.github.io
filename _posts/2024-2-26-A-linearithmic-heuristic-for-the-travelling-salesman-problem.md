---
layout:     post
title:      A linearithmic heuristic for the travelling salesman problem
subtitle:   EJOR2021 分治求解TSP
date:       2024/2/26
author:     Birdie
header-img: img/post_header_sr.jpg
catalog: true
tags:
    - 论文阅读
    - 组合优化
    - EJOR
---

A linearithmic heuristic for the travelling salesman problem

European Journal of Operational Research 2021

University of Applied Sciences of Western Switzerland



## 方法

![image-20240226144601624]({{site.url}}/img/2024-2-26-A-linearithmic-heuristic-for-the-travelling-salesman-problem/image-20240226144601624.png)

### RecorderPath

![image-20240226144637822]({{site.url}}/img/2024-2-26-A-linearithmic-heuristic-for-the-travelling-salesman-problem/image-20240226144637822.png)

- 输入：$P=(b=c_i,c_1,\cdots,c_{i-1},c_{i+1},\cdots,c_n,c_i=e)$，超参数 $t$，一般取 10-20。

- 需要：OptimisePath，能够得到一条最优的路径（已知起点和终点），至少需要百级的



1. 如果 $P$ 的规模小于 $t^2$​，直接放进 OptimisePath 能够得到最优解。
2. 找到距离 $b$ 最近的节点 $u$，找到距离 $e$ 最近的节点 $v$，然后再剩下的节点中随机挑出 $t-2$ 个。
3. 将 $t-2$ 个节点和 $u,v$ 放进 OptimisePath，得到最优路径 $P_S=(b,s_1,\cdots,s_t,e)$。
4. 然后对于在 $P$ 中但是不在 $P_S$ 中的节点，找到在 $P_S$ 中最接近的一个节点，放在他后面。
5. 现在得到了 $P1=(b=s_1,\cdots,s_2),P2=(b=s_2,\cdots,s_3),\cdots$​，对每一段分别再做一次 RecorderPath。



![image-20240226150315395]({{site.url}}/img/2024-2-26-A-linearithmic-heuristic-for-the-travelling-salesman-problem/image-20240226150315395.png)



### FastPopmusic

对 RecorderPath 得到的路径 $P$，每隔 $t^2$ 个就重新做一次 RecorderPath ，直到没得更新为止。

![image-20240226150328261]({{site.url}}/img/2024-2-26-A-linearithmic-heuristic-for-the-travelling-salesman-problem/image-20240226150328261.png)