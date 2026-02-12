Self-Play Preference Optimization For Language Model Alignment

加利福利亚大学洛杉矶分校、CMU

## 摘要

标准的基于人类反馈的强化学习（RLHF）方法依赖于参数化模型（如Bradley-Terry模型），在捕捉人类偏好的非传递性和非理性方面存在不足。近期研究表明，直接处理偏好概率可以更准确地反映人类偏好，从而实现更灵活、更精确的语言模型对齐。本文提出了一种基于自博弈的语言模型对齐方法，将该问题视为一个旨在识别纳什均衡策略的常数和双人博弈。我们的方法称为自博弈偏好优化（Self-Play Preference Optimization, SPPO），利用迭代策略更新来可证明地近似纳什均衡。此外，我们提出了一种新的SPPO目标函数，该函数既有充分的理论动机，在实践中又简单有效。在实验中，仅使用来自UltraFeedback数据集的60k提示（不含回复）且无需任何提示增强，通过利用仅有0.4B参数的预训练偏好模型PairRM，SPPO可以从Mistral-7B-Instruct-v0.2的微调中获得一个模型，该模型在AlpacaEval 2.0上相对于GPT-4-Turbo达到28.53%的最先进的长度控制胜率。它在MT-Bench、Arena-Hard和Open LLM Leaderboard上也优于（迭代）DPO和IPO。从更强的基础模型Llama-3-8B-Instruct出发，我们能够达到38.77%的长度控制胜率。值得注意的是，SPPO的强劲表现是在没有来自GPT-4或其他更强语言模型的额外外部监督（例如回复、偏好等）的情况下实现的。

## 引入

受到自博弈偏好优化（Self-play Preference Optimization, SPO）的启发，提出了一种新的自博弈算法。

该算法

（1）具有可证明的保证来求解双人常数和博弈；

（2）能够扩展到大规模LLM的高效微调。

具体而言，我们将RLHF问题表述为常数和双人博弈。我们的目标是识别纳什均衡策略，该策略平均而言始终提供优于任何其他策略的回复。为近似识别纳什均衡策略，我们采用经典的带乘法权重的在线自适应算法作为求解双人博弈的高层框架。此外，高层框架的每一步都可以通过自博弈机制来近似，其中在每次迭代中，策略通过在前一次迭代的策略上微调自身来与自身博弈，微调数据由策略生成并由偏好模型标注。

我们的贡献突出如下：

- 从可证明收敛到双人常数和博弈纳什均衡的指数权重更新算法出发，我们提出了用于大型语言模型对齐的自博弈偏好优化（SPPO）算法。该算法可证明地收敛到近似纳什均衡，并允许简单形式的损失函数以便于优化。
- 与DPO和身份偏好优化（Identity Preference Optimization, IPO）等对称成对损失不同，我们提出了一种不依赖于成对比较的新优化目标。新的损失目标最初由博弈论概念驱动，结果证明与策略梯度理论有强烈动机联系，并隐式鼓励LLM学习token级最优价值函数。

## 预备知识

我们考虑如下偏好学习场景。给定一个文本序列（通常称为提示）$\mathbf{x} = [x\_1, x\_2, \dots]$，生成两个文本序列 $\mathbf{y} = [y\_1, y\_2, \dots]$ 和 $\mathbf{y}'$ 作为对提示 $\mathbf{x}$ 的回复。一个自回归语言模型 $\pi$ 在给定提示 $\mathbf{x}$ 的情况下，可以按照如下概率分解生成回复 $\mathbf{y}$：



$$

\pi(\mathbf{y} \mid \mathbf{x}) = \prod_{i=1}^{N} \pi(y_i \mid \mathbf{x}, \mathbf{y}_{<i}).

$$



给定提示 $\mathbf{x}$ 和两个回复 $\mathbf{y}$ 和 $\mathbf{y}'$，偏好oracle（无论是人类标注者还是语言模型）将提供偏好反馈 $o(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x}) \in \lbrace 0, 1 \rbrace $，指示 $\mathbf{y}$ 是否优于 $\mathbf{y}'$。我们记 $\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x}) = \mathbb{E}[o(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})]$ 为 $\mathbf{y}$ "赢得对决"相对于 $\mathbf{y}'$ 的概率。两个概率分布 $p$ 和 $q$ 的KL散度定义为：



$$

\mathrm{KL}(p\ \mid q) = \mathbb{E}_{\mathbf{y} \sim p(\mathbf{y})}\left[\log \frac{p(\mathbf{y})}{q(\mathbf{y})}\right].

$$



### 基于奖励模型的RLHF

Christiano首先按照Bradley-Terry模型学习奖励函数 $r(\mathbf{y}; \mathbf{x})$。对于一个提示-回复-回复三元组 $(\mathbf{x}, \mathbf{y}, \mathbf{y}')$，Bradley-Terry模型指定 $\mathbf{y}$ 被选为优于 $\mathbf{y}'$ 的概率为：



$$

\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x}) = \frac{\exp(r(\mathbf{y}; \mathbf{x}))}{\exp(r(\mathbf{y}; \mathbf{x})) + \exp(r(\mathbf{y}'; \mathbf{x}))} = \sigma\left(r(\mathbf{y}; \mathbf{x}) - r(\mathbf{y}'; \mathbf{x})\right),

$$



其中 $\sigma(x) = e^x / (e^x + 1)$ 是logistic函数。与Bradley-Terry模型相关的奖励函数可以通过最大化对数似然 $\log \mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})$ 来估计。假设真实的奖励函数 $r(\mathbf{y}; \mathbf{x})$ 可用，Christiano提出使用RL中的策略优化算法（如PPO）求解如下优化问题：



$$

\max_{\theta} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}[r(\mathbf{y}; \mathbf{x})] - \eta^{-1} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}}[\mathrm{KL}(\pi_{\theta}(\cdot \mid \mathbf{x}) \ \mid \pi_{\text{ref}}(\cdot \mid \mathbf{x}))],

$$



其中 $\mathcal{X}$ 是提示分布。

Rafailov发现上述优化问题具有闭式解，使得对于任意 $\mathbf{y}$，



$$

\pi^\ast (\mathbf{y} \mid \mathbf{x}) \propto \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp(\eta r(\mathbf{y}; \mathbf{x})),

$$



这可以进一步转换为对于任意三元组 $(\mathbf{x}, \mathbf{y}\_w, \mathbf{y}\_l)$ 的DPO损失，其中胜者 $\mathbf{y}_w$ 被选为优于败者 $\mathbf{y}_l$：



$$

\ell_{\text{DPO}}(\mathbf{x}, \mathbf{y}_w, \mathbf{y}_l; \theta; \pi_{\text{ref}}) := -\log \sigma\left(\eta^{-1}\left[\log\left(\frac{\pi_{\theta}(\mathbf{y}_w \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_w \mid \mathbf{x})}\right) - \log\left(\frac{\pi_{\theta}(\mathbf{y}_l \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l \mid \mathbf{x})}\right)\right]\right).

$$



### 基于一般偏好的RLHF

我们旨在建立无需奖励模型的RLHF方法，因为人类偏好可能是非传递的。在一般偏好oracle $\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})$ 下，旨在识别von Neumann赢家。更具体地说，von Neumann赢家 $\pi^\ast $ 是如下双人常数和博弈的（对称）纳什均衡：



$$

(\pi^\ast , \pi^\ast ) = \arg\max_{\pi} \min_{\pi'} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}}\left[\mathbb{E}_{\mathbf{y} \sim \pi(\cdot \mid \mathbf{x}), \mathbf{y}' \sim \pi'(\cdot \mid \mathbf{x})}\left[\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})\right]\right].

$$



> \ast \ast 偏好Oracle P\ast \ast = 判断好坏的外部裁判（可以是人类、GPT-4、规则等）

此外，我们定义一个回复 $\mathbf{y}$ 相对于回复分布 $\pi$ 的获胜概率为：


$$

\mathbb{P}(\mathbf{y} \succ \pi \mid \mathbf{x}) = \mathbb{E}_{\mathbf{y}' \sim \pi(\cdot \mid \mathbf{x})}[\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})],

$$



以及一个策略 $\pi$ 相对于另一个策略 $\pi'$ 的获胜概率为：



$$

\mathbb{P}(\pi \succ \pi' \mid \mathbf{x}) = \mathbb{E}_{\mathbf{y} \sim \pi(\cdot \mid \mathbf{x})}\mathbb{E}_{\mathbf{y}' \sim \pi'(\cdot \mid \mathbf{x})}[\mathbb{P}(\mathbf{y} \succ \mathbf{y}' \mid \mathbf{x})].

$$



此外，我们定义 $\mathbb{P}(\pi \succ \pi') = \mathbb{E}_{\mathbf{x} \sim \mathcal{X}}[\mathbb{P}(\pi \succ \pi' \mid \mathbf{x})]$，其中 $\mathbf{x}$ 是从提示分布 $\mathcal{X}$ 中抽取的提示。双人常数和博弈可以简化为：



$$

(\pi^\ast , \pi^\ast ) = \arg\max_{\pi} \min_{\pi'} \mathbb{P}(\pi \succ \pi').

$$



> 通俗解释：
>
> 传统的RLHF（基于人类反馈的强化学习）通常假设：
> - 如果人类喜欢A胜过B，喜欢B胜过C，那么一定喜欢A胜过C
> - 这种"传递性"在现实中经常不成立（比如你今天觉得A好，明天又觉得C好）
>
> 想象一个\ast \ast 石头剪刀布\ast \ast 的游戏：
> - 没有绝对的"最强"，只有相互克制
> - 但如果你能找到一种\ast \ast 混合策略\ast \ast （比如随机出石头、剪刀、布），让对手无论怎么选都无法长期击败你，这就是"纳什均衡"
>
> \ast \ast von Neumann赢家\ast \ast 就是这个概念在RLHF中的对应物：找到一个AI策略 $\pi^\ast $，使得\ast \ast 无论对手用什么策略 $\pi'$，我至少能打成平手或获胜\ast \ast
>
> 用数学表达就是：
>

$$

> \pi^\ast = \arg\max_{\pi} \min_{\pi'} \mathbb{P}(\pi \succ \pi')
>

$$


>
> 意思是：\ast \ast 我选择策略 $\pi$，然后对手选择最针对我的策略 $\pi'$，我要最大化这个最坏情况下的胜率\ast \ast
>
> 直观理解
>
> 想象你在训练一个\ast \ast 辩论AI\ast \ast ：
> - 传统方法：找一个"评分员"给每个回答打分，然后优化分数
> - 新方法：让AI直接和另一个AI"对战"，人类当裁判说谁赢。目标是找到一个AI，\ast \ast 无论对手是谁，人类都更倾向于选我\ast \ast
>
> 这就像训练一个\ast \ast 全能选手\ast \ast ——不是在某一方面最强，而是\ast \ast 没有明显弱点\ast \ast ，面对任何对手都能至少不落下风。
>
> 公式中的 $(\pi^\ast , \pi^\ast )$ 表示\ast \ast 自己打自己\ast \ast 也能达到平衡：
>- 左 $\pi^\ast $：我要最大化我的胜率
> - 右 $\pi^\ast $：对手（也是同样的我）要最小化我的胜率
>
> 当两边都是同一个 $\pi^\ast $ 时达到均衡，意味着这个策略\ast \ast 对自己也是最优的\ast \ast ——没有"自我矛盾"。
>

## 方法

本节介绍自博弈偏好优化（Self-Play Preference Optimization, SPPO）算法，该算法源自以下理论框架。

### 理论框架

存在求解常数和双人博弈中纳什均衡的知名算法。在本工作中，我们建立一个迭代框架，可以渐近地收敛到平均意义上的最优策略。我们从概念上求解双人博弈的理论框架开始：



$$

\pi_{t+1}(\mathbf{y} \mid \mathbf{x}) \propto \pi_t(\mathbf{y} \mid \mathbf{x}) \exp(\eta \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x})), \quad \text{for } t = 1, 2, \ldots \tag{3.1}

$$



(3.1)是一个迭代框架，在每次迭代$t$中依赖乘法权重更新，并具有清晰的结构。最初，我们有一个来自某些监督微调模型的基础策略$\pi_1$。在每次迭代中，更新后的策略$\pi_{t+1}$从参考策略$\pi_t$通过乘法权重更新获得。更具体地说，如果一个回复$\mathbf{y}$相对于当前策略$\pi_t$具有更高的平均优势，则它应该具有更高的概率权重。

>

$$

> \pi_{t+1}(\mathbf{y} \mid \mathbf{x}) \propto \pi_t(\mathbf{y} \mid \mathbf{x}) \exp(\eta \cdot \text{胜率})
>

$$


>
> \ast \ast 通俗理解\ast \ast ：
>
> - $\pi_t$ 是第 $t$ 轮的"学生水平"
> - 对于同一个题目 $\mathbf{x}$，学生写出了答案 $\mathbf{y}$
> - 我们让 $\mathbf{y}$ 与\ast \ast 当前水平的自己\ast \ast （$\pi_t$）比赛
> - 如果 $\mathbf{y}$ 的\ast \ast 胜率\ast \ast 高，就在下一轮给它\ast \ast 更高的权重\ast \ast （乘以一个大于1的数）
> - 如果胜率低，就给更低权重
>

等价地，(3.1)可以写成：



$$

\pi_{t+1}(\mathbf{y} \mid \mathbf{x}) = \frac{\pi_t(\mathbf{y} \mid \mathbf{x}) \exp\left(\eta \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x})\right)}{Z_{\pi_t}(\mathbf{x})}, \tag{3.2}

$$



其中$Z\_{\pi\_t}(\mathbf{x}) = \sum\_{\mathbf{y}} \pi\_t(\mathbf{y} \mid \mathbf{x}) \exp\left(\eta \mathbb{P}(\mathbf{y} \succ \pi\_t \mid \mathbf{x})\right)$是归一化因子（也称为配分函数）。对于任何固定的$\mathbf{x}$和$\mathbf{y}$，理想的更新策略$\pi_{t+1}$应满足以下方程：



$$

\log\left(\frac{\pi_{t+1}(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) = \eta \cdot \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x}) - \log Z_{\pi_t}(\mathbf{x}). \tag{3.3}

$$



与DPO或IPO中通过对$\mathbf{y}$和$\mathbf{y}'$之间的(3.3)求差来消去对数归一化因子$\log Z\_{\pi\_t}(\mathbf{x})$的成对设计不同，我们选择直接以L2距离近似(3.3)：



$$

\pi_{t+1} = \arg\min_{\pi} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_t(\cdot \mid \mathbf{x})}\left(\log\left(\frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) - \left(\eta \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x}) - \log Z_{\pi_t}(\mathbf{x})\right)\right)^2. \tag{3.4}

$$



> 理想情况下，我们希望新策略满足：
>

$$

> \log(\text{新策略}/\text{旧策略}) = \eta \cdot \text{胜率} - \text{归一化项}
>

$$


>
> 但直接算那个复杂的\ast \ast 归一化因子\ast \ast $\log Z$ 很麻烦（要遍历所有可能的答案）。
>
> \ast \ast 聪明的做法\ast \ast ：不精确求解，而是用\ast \ast 最小二乘法\ast \ast 近似——让左边尽量接近右边（3.4中），允许有点小误差。这就像用直线拟合曲线，牺牲一点精度换取计算可行性。

\ast \ast 概率估计\ast \ast 优化目标(3.4)可以用有限样本近似。我们选择为每个提示$\mathbf{x}$采样$K$个回复$\mathbf{y}\_1, \mathbf{y}\_2, \ldots, \mathbf{y}\_K \sim \pi\_t(\cdot \mid \mathbf{x})$，并将经验分布记为$\widehat{\pi}_t^K$。有限样本优化问题可以近似为：


$$

\pi_{t+1} = \arg\min_{\pi} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_t(\cdot \mid \mathbf{x})}\left(\log\left(\frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) - \left(\eta \mathbb{P}(\mathbf{y} \succ \widehat{\pi}_t^K \mid \mathbf{x}) - \log Z_{\widehat{\pi}_t^K}(\mathbf{x})\right)\right)^2. \tag{3.5}

$$



具体而言，$\mathbb{P}(\mathbf{y} \succ \widehat{\pi}\_t^K \mid \mathbf{x}) = \sum\_{k=1}^{K} \mathbb{P}(\mathbf{y} \succ \mathbf{y}\_k \mid \mathbf{x})/K$，且



$$

Z_{\widehat{\pi}_t^K}(\mathbf{x}) = \mathbb{E}_{\mathbf{y} \sim \pi_t(\cdot \mid \mathbf{x})}[\exp(\eta \mathbb{P}(\mathbf{y} \succ \widehat{\pi}_t^K \mid \mathbf{x}))].

$$



$Z\_{\widehat{\pi}\_t^K}(\mathbf{x})$，被视为一个期望，可以进一步用$B$个新样本估计，总共需要$O(KB)$次偏好oracle $\mathbb{P}$的查询。

(3.5)是一个可有效处理的优化问题。非正式地说，当$K \to \infty$时，(3.5)将恢复(3.4)。我们对(3.4)的收敛性有以下保证：

\ast \ast 定理3.1\ast \ast 假设优化问题(3.4)是可实现的。将通过(3.4)获得的策略记为$\pi_t$，混合策略$\bar{\pi}\_T = \frac{1}{T}\sum\_{t=1}^{T} \pi\_t$。通过设置$\eta = \Theta(1/\sqrt{T})$，我们有：



$$

\max_{\pi}[\mathbb{P}(\pi \succ \bar{\pi}_T)] - \min_{\pi}[\mathbb{P}(\pi \prec \bar{\pi}_T)] = O(1/\sqrt{T}).

$$



定理3.1以对偶间隙的形式刻画了时间范围$T$内平均策略向纳什均衡收敛的速率。

或者，我们可以通过基于人类偏好模型用常数替换$\log Z\_{\widehat{\pi}\_t^K}(\mathbf{x})$来避免估计它。该常数的选择在附录E中详细讨论。这里，我们在(3.5)中用$\eta/2$替换$\log Z\_{\widehat{\pi}\_t^K}(\mathbf{x})$以获得更清晰的目标：



$$

\pi_{t+1} = \arg\min_{\pi} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_t(\cdot \mid \mathbf{x})}\left(\log\left(\frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) - \eta\left(\mathbb{P}(\mathbf{y} \succ \widehat{\pi}_t^K \mid \mathbf{x}) - \frac{1}{2}\right)\right)^2. \tag{3.6}

$$



直观上，如果出现平局（即$\mathbb{P}(\mathbf{y} \succ \widehat{\pi}_t^K \mid \mathbf{x}) = 1/2$），我们希望模型不在$\mathbf{y}$处更新权重。如果$\mathbf{y}$平均而言战胜$\widehat{\pi}_t^K$（即$\mathbb{P}(\mathbf{y} \succ \widehat{\pi}_t^K \mid \mathbf{x}) > 1/2$），则我们增加$\mathbf{y}$处的概率密度以利用$\mathbf{y}$相对于$\widehat{\pi}_t^K$的优势。在我们的实验中，我们选择最小化目标(3.6)。

> \ast \ast 问题\ast \ast ：真实的胜率需要对无穷多个答案求平均，算不了。
>
> \ast \ast 解决方案\ast \ast ：
> 1. \ast \ast 采样估计\ast \ast ：每个题目只采样 $K$ 个答案来估算胜率（用样本代替总体）
> 2. \ast \ast 避免算归一化项\ast \ast ：直接把那个麻烦的常数替换成 $\eta/2$
>
> 最终得到实用的目标函数（3.6）：
>
>

$$

> \min \mathbb{E}\left[\left(\log\frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})} - \eta\left(\text{胜率} - \frac{1}{2}\right)\right)^2\right]
>

$$


>
> \ast \ast 直观含义\ast \ast ：
> - \ast \ast 胜率 = 50%\ast \ast （平局）→ 不更新（减1/2后为0）
> - \ast \ast 胜率 > 50%\ast \ast （赢了）→ 增加这个答案的概率（正向更新）
> - \ast \ast 胜率 < 50%\ast \ast （输了）→ 减少这个答案的概率（负向更新）
>

### SPPO算法

基于上述理论框架，我们在算法1中提出了自博弈偏好优化算法。

\ast \ast 算法1\ast \ast 自博弈偏好优化（SPPO）

\ast \ast 输入\ast \ast ：基础策略$\pi\_{\theta\_1}$，偏好oracle $\mathbb{P}$，学习率$\eta$，生成样本数$K$

\ast \ast for\ast \ast $t = 1, 2, \ldots$ \ast \ast do\ast \ast

1. 通过采样$\mathbf{x} \sim \mathcal{X}$和$\mathbf{y}\_{1:K} \sim \pi\_t(\cdot \mid \mathbf{x})$生成合成回复
2. 标注胜率$\mathbb{P}(\mathbf{y}\_k \succ \mathbf{y}\_{k'} \mid \mathbf{x}), \forall k, k' \in [K]$
3. 从$\mathbf{y}_{1:K}$中选择回复构建数据集$\mathcal{D}\_t = \lbrace (\mathbf{x}\_i, \mathbf{y}\_i, \widehat{P}(\mathbf{y}\_i \succ \pi\_t \mid \mathbf{x}\_i)) \rbrace \_{i \in [N]}$
4. 根据(3.6)优化$\pi\_{\theta\_{t+1}}$：



$$

\theta_{t+1} \leftarrow \arg\min_{\theta} \mathbb{E}_{(\mathbf{x}, \mathbf{y}, \widehat{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x})) \sim \mathcal{D}_t}\left(\log\left(\frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) - \eta\left(\widehat{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x}) - \frac{1}{2}\right)\right)^2 \tag{3.7}

$$



\ast \ast end for\ast \ast

在每次迭代$t$中，算法1将首先根据$\pi_t(\cdot \mid \mathbf{x})$为每个提示$\mathbf{x}$生成$K$个回复$\mathbf{y}\_1, \mathbf{y}\_2, \ldots, \mathbf{y}\_K$。然后，将查询偏好oracle $\mathbb{P}$以计算$K$个回复之间的胜率。在第5行，可以应用某些标准来确定哪些回复应保留在构建的数据集$\mathcal{D}_t$中，并构建提示-回复-概率三元组$(\mathbf{x}, \mathbf{y}, \widehat{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x}))$。一个直接的设计选择是将所有$K$个回复包含到$\mathcal{D}_t$中，每个$\widehat{P}(\mathbf{y}\_i \succ \pi\_t \mid \mathbf{x})$通过与所有$K$个回复比较来估计。总共将进行$O(K^2)$次查询。然后算法将在数据集$\mathcal{D}_t$上优化(3.6)。

### 与策略梯度的联系

虽然SPPO源自双人博弈的迭代框架，但SPPO目标(3.4)中的平方损失由于其二乘损失形式，为SPPO作为策略梯度方法的半在线变体提供了另一种解释。与标准策略梯度的区别在于，它在迭代$t$开始时从$\pi\_{\theta\_t}$收集样本，而不是在每个梯度步骤执行在线采样。

考虑一般奖励函数$r(\mathbf{y}; \mathbf{x})$，RLHF问题(2.2)可以写成：



$$

\max_{\theta} J(\theta) := \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}\left[r(\mathbf{y}; \mathbf{x}) - \eta^{-1} \log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}\right].

$$



目标$J(\theta)$的策略梯度为：



$$

\nabla J(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}\left[\left(r(\mathbf{y}; \mathbf{x}) - \eta^{-1} \log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} - b(\mathbf{x})\right) \nabla \log \pi_{\theta}(\mathbf{y} \mid \mathbf{x})\right]

$$





$$

= \frac{\eta}{2} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}\left[-\nabla\left(r(\mathbf{y}; \mathbf{x}) - \eta^{-1} \log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} - b(\mathbf{x})\right)^2\right], \tag{3.10}

$$



其中第一行遵循策略梯度定理，基线$b(\mathbf{x})$是仅依赖于$\mathbf{x}$的任意常数，用于方差缩减。将平方损失(3.10)与SPPO目标(3.4)（重写如下）比较：



$$

\theta_{t+1} = \arg\min_{\theta} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta_t}(\cdot \mid \mathbf{x})}\left[\left(\mathbb{P}(\mathbf{y} \succ \pi_{\theta_t} \mid \mathbf{x}) - \eta^{-1} \log\left(\frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{\theta_t}(\mathbf{y} \mid \mathbf{x})}\right) - \eta^{-1} \log Z_{\pi_{\theta_t}}(\mathbf{x})\right)^2\right],

$$



可以看出，胜率$\mathbb{P}(\mathbf{y} \succ \pi\_{\theta\_t} \mid \mathbf{x})$正是SPPO旨在最大化的奖励，而$\eta^{-1} \log Z\_{\pi\_{\theta\_t}}(\mathbf{x})$实际上是最佳可能的基线——（软）价值函数。当实践中价值函数不可用时，可以用任何常数基线替换以减少策略梯度的方差。我们选择$1/2$作为$\eta^{-1} \log Z\_{\pi\_{\theta\_t}}(\mathbf{x})$的良好近似，但该常数可能因人类偏好模型而异。

SPPO可以被视为一种新的、直接的策略梯度方法变体，无需诸如PPO中的梯度裁剪、TRPO中的Hessian计算或许多策略优化算法中维护多个组件（Q-critic、V-critic、actor等）等额外修改。

### Token级$Q^\ast $学习

在最大熵RL公式下，token级对数比率$\log \frac{\pi\_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi\_{\text{ref}}(\mathbf{y} \mid \mathbf{x})}$可以被视为隐式token级奖励或优势函数（在奖励整形下不变）。下面我们展示SPPO中的平方损失也可以导致最优最大熵策略$\pi^\ast $，具有token级最优价值/优势函数。

Token级MDP将状态$\mathbf{s}\_h = (\mathbf{x}, y\_1, y\_2, \ldots, y\_{h-1})$定义为前缀token，动作$\mathbf{a}\_h = y\_h$定义为下一个token。自回归语言模型$\pi(\mathbf{y} \mid \mathbf{x})$可以被视为token级策略$\pi(\mathbf{a}\_h \mid \mathbf{s}\_h)$，转移核是已知且确定性的，因为它只是将下一个token连接到前缀以形成新的token序列$\mathbf{s}\_{h+1} = (\mathbf{x}, y\_1, y\_2, \ldots, y\_h)$。

最大熵RL设置再次考虑反向KL正则化奖励最大化问题(2.2)：



$$

\max_{\theta} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}[r(\mathbf{y}; \mathbf{x})] - \eta^{-1} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}}[\mathrm{KL}(\pi_{\theta}(\cdot \mid \mathbf{x}) \ \mid \pi_{\text{ref}}(\cdot \mid \mathbf{x}))]

$$





$$

= -\mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_{\theta}(\cdot \mid \mathbf{x})}[r(\mathbf{y}; \mathbf{x}) + \eta^{-1} \log \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})] + \eta^{-1} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}}[\mathcal{H}(\pi_{\theta}(\cdot \mid \mathbf{x}))].

$$



我们将上述问题的最优解记为$\pi^\ast $。Rafailov et al. (2024a)表明Bradley-Terry偏好模型(B.2)可以重写为：



$$

\mathbb{P}(\mathbf{y}_w \succ \mathbf{y}_l \mid \mathbf{x}) = \sigma\left(\eta^{-1} \sum_{h=1}^{ \mid \mathbf{y}_w \mid } \log \frac{\pi^\ast (\mathbf{a}_h^w \mid \mathbf{s}_h^w)}{\pi_{\text{ref}}(\mathbf{a}_h^w \mid \mathbf{s}_h^w)} - \eta^{-1} \sum_{h=1}^{ \mid \mathbf{y}_l \mid } \log \frac{\pi^\ast (\mathbf{a}_h^l \mid \mathbf{s}_h^l)}{\pi_{\text{ref}}(\mathbf{a}_h^l \mid \mathbf{s}_h^l)}\right),

$$



其中状态和动作定义如上所述的token级MDP，上标$(\cdot)^w$和$(\cdot)^l$表示是用于胜者$\mathbf{y}_w$还是败者$\mathbf{y}_l$。用$\pi_{\theta}$替换$\pi^\ast $最大化对数似然即得到DPO损失。

从现在起，为简单起见，我们假设时间范围为固定的$H$。最大熵RL公式的推导依赖于（软）最优价值函数$Q^\ast $和$V^\ast $：



$$

V^\ast (\mathbf{s}_{H+1}) = r(\mathbf{s}_{H+1}) := r(\mathbf{y}; \mathbf{x}), \quad \text{(EOS处的奖励)}

$$





$$

Q^\ast (\mathbf{s}_h, \mathbf{a}_h) = \eta^{-1} \log \pi_{\text{ref}}(\mathbf{a}_h \mid \mathbf{s}_h) + V^\ast (\mathbf{s}_{h+1}),

$$





$$

V^\ast (\mathbf{s}_h) = \eta^{-1} \log \sum_{\mathbf{a}} \exp\left(\eta Q^\ast (\mathbf{s}_h, \mathbf{a})\right), \quad \text{对于 } h \leq H.

$$



最优策略$\pi^\ast $满足：



$$

\eta^{-1} \log \pi^\ast (\mathbf{a}_h \mid \mathbf{s}_h) = Q^\ast (\mathbf{s}_h, \mathbf{a}_h) - V^\ast (\mathbf{s}_h)

$$





$$

= \eta^{-1} \log \pi_{\text{ref}}(\mathbf{a}_h \mid \mathbf{s}_h) + V^\ast (\mathbf{s}_{h+1}) - V^\ast (\mathbf{s}_h).

$$



可以验证，对于$\mathbf{s}_1 = (\mathbf{x})$，我们有$\eta V^\ast (\mathbf{s}\_1) = \log \sum\_{\mathbf{y}} \pi\_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp(\eta r(\mathbf{y}; \mathbf{x}))$。回到第$t$次迭代的SPPO目标(3.4)，如果我们设置$\pi\_{\text{ref}} = \pi\_t$和$r(\mathbf{y}; \mathbf{x}) = \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x})$，我们有$V^\ast (\mathbf{s}\_1) = \eta^{-1} \log Z\_{\pi\_t}(\mathbf{x})$，第$t$次迭代的学习目标变为：



$$

\pi_{t+1} = \arg\min_{\pi} \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, \mathbf{y} \sim \pi_t(\cdot \mid \mathbf{x})}\left(\log\left(\frac{\pi(\mathbf{y} \mid \mathbf{x})}{\pi_t(\mathbf{y} \mid \mathbf{x})}\right) - \left(\eta \mathbb{P}(\mathbf{y} \succ \pi_t \mid \mathbf{x}) - \log Z_{\pi_t}(\mathbf{x})\right)\right)^2

$$





$$

= \arg\min_{\pi} \mathbb{E}_{\mathbf{s}_1 \sim \mathcal{X}, \mathbf{a}_h \sim \pi_t(\cdot \mid \mathbf{s}_h)}\left(\sum_{h=1}^{H} \log \frac{\pi(\mathbf{a}_h \mid \mathbf{s}_h)}{\pi^\ast (\mathbf{a}_h \mid \mathbf{s}_h)}\right)^2. \tag{3.11}

$$



与DPO类似，SPPO通过平方损失形式(3.11)"秘密地"鼓励策略$\pi_{\theta}$在token级收敛到最优策略$\pi^\ast $。此外，人们可能意识到，最小化平方损失形式与通过策略梯度最小化KL散度$\mathrm{KL}(\pi_{\theta} \ \mid \pi^\ast )$相关：



$$

\nabla_{\theta} \mathrm{KL}(\pi_{\theta} \ \mid \pi^\ast ) = \mathbb{E}_{\mathbf{s}_1 \sim \mathcal{X}, \mathbf{a}_h \sim \pi_{\theta}(\cdot \mid \mathbf{s}_h)}\left[\left(\sum_{h=1}^{H} \log \frac{\pi_{\theta}(\mathbf{a}_h \mid \mathbf{s}_h)}{\pi^\ast (\mathbf{a}_h \mid \mathbf{s}_h)}\right) \sum_{h=1}^{H} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_h \mid \mathbf{s}_h)\right]

$$





$$

= \mathbb{E}_{\mathbf{s}_1 \sim \mathcal{X}, \mathbf{a}_h \sim \pi_{\theta}(\cdot \mid \mathbf{s}_h)}\left[\nabla_{\theta}\left(\sum_{h=1}^{H} \log \frac{\pi_{\theta}(\mathbf{a}_h \mid \mathbf{s}_h)}{\pi^\ast (\mathbf{a}_h \mid \mathbf{s}_h)}\right)^2\right].

$$



## 实验

不是这个领域的，所以没关注。



## 总结

\ast \ast 局限性\ast \ast 理论上，通过回归近似最优策略依赖于模型类具有足够表达能力和生成数据良好覆盖输入空间的假设。用常数近似对数配分因子仅在接近软价值函数时有助于减少方差。实验在单个数据集UltraFeedback上运行，由于计算资源有限，模型仅在少数基准上测试，但如果有更多，所提出的方法可以在更多模型、数据集和基准上进一步验证以获得全面评估。






