---
layout:     post
title:      Virginia加密
subtitle:   信息安全技术
date:       2022/5/02
author:     Birdie
header-img: img/post_header.jpg
catalog: true
tags:
    - 信息安全
---

### 零   实验环境

​		操作系统：`Windows 10`

​		程序语言：`C++`

​		编译环境：`g++ (MinGW-W64) 8.1.0`



### 一   实验目标

​		实现维吉尼亚密码的破译流程，并破译附件 txt 文件中的一段由维吉尼亚密码加密得到的密文。



### 二   分析过程

#### 加密原理

​		维吉尼亚密码是m维向量形式的移位密码。假设串长为`L`，明文为`P`，密文为`C`，密钥为`K`，则：

$$
C=(P_1+K_1,P_2+K_2,...,P_L+K_L)mod\ 26 \\
P=(P_1-K_1,P_2-K_2,...,P_L-K_L)mod\ 26
$$


#### 解密原理

##### 概述

​		首先，维吉尼亚密码是多表代换密码，对于维吉尼亚密码中的 m 个表有对应 m 个密钥，我们不妨设为 0 ~ m - 1 号密钥。对于密文中的第 i 个字符，我们使用第 i % m 号密钥进行移位代换加密。虽然对于每个表都是简单的移位密码，但当 m 未知的时候整段密文还是非常混乱的。

​		所以要想解密维吉尼亚密码，首先需要找到密钥的长度 m。我们首先使用 Kasiski 测试法通过找到若干个有一定长度的相同密文段，找到它们距离的最大公因子 q，那我们就有理由猜测我们最后的 m 可能是 q 的因子。为了验证这个猜想，我们枚举 q 的每个因子进行重合指数法。我们知道对于一个英文文本串，出现 A ~ Z 的期望概率平方的累加和大概是 0.065，而完全随机的文本串确只有 0.038。由于我们对于每个表都只是简单的移位密码，所以加密前后 A ~ Z 的期望概率平方的累加和是不会变的。所以我们如果发现 q 的一个因子让我们按照这个因子分组后每一组 A ~ Z 的期望概率平方的累加和接近 0.065，我们就有理由猜测这是有意义的明文加密来的，进而猜测就是我们的表数 m。最后我们就只要枚举每个表的密钥，看看移位多少能让这一组的期望值接近于 A ~ Z 的概率统计分布就可以了。

##### Kasiski测试

​		因为若用给定的m个密钥表周期地对明文字母进行加密，则当明文中有两个相同的字母组（长度大于3）在明文序列中间隔得字母数为m的倍数时，这两个明文字母组对应的密文字母组必相同。但反过来，若密文中出现两个相同的字母组，它们所对应的明文字母组未必相同，但相同的可能性极大。所以我们可以在密文中寻找重复次数>=3的字母组。然后计算重复字母组的两两相邻的距离差。

##### 重合指数法

​		设一门语言由 n 个字母构成，每个字母出现的概率为$P_i(1\leq i\leq n)$，则重合指数是指其中两个随机元素相同的概率的和，即为$CI=\sum_{i=1}^{n}P_i^2$​。完全随机的英文文本的重合指数为0.0385 ，一个有意义的英文文本则为0.065。实际使用 CI 的估计值为$CI'$，$L$为密文的长度，$f_i$为26个字母中第$i$个字母发生的次数。

$$
CI'=\sum_{i=1}^{n}\frac{f_i}{L}\cdot \frac{f_i-1}{L-1}
$$

​		$CI'$是能够区分单表代换密码和多表代换密码。如果密文的重合指数较低，就可能是一个多表代换密码。因为单表代换改变的只是字母，并没有改变频率，故统计规律和自然语言相似。若结果接近0.065，则猜测正确。

#### 解密过程

##### 步骤一：求出密钥的长度

​		首先我们还是要先计算出密钥的长度 m。要实现这个目标，我们一共要进行三步。

###### 第一步： Kasiski 测试法

​		第一步是通过 Kasiski 测试法，我们先统计长度为 3 出现次数最多的子串，这个可以通过从左到右枚举所有长度为 3 的子串，丢到 map 里面，不停更新出现次数最多的子串来实现。我们输出这一步答案发现 CHR 出现了 5 次。下面是这部分代码：

```c++
/*
maxStr : 出现次数最多的长度为3的子串
maxNum : 出现次数
*/
map <string, int> mp;
int L = s.length();
int maxNum = 0; string maxStr = "";
for (int i = 0; i < L - 2; i++) {
    string now = "";
    for (int j = 0; j < cn; j++) now += s[i + j];
    mp[now]++;
    if (mp[now] > maxNum) {
        maxNum = mp[now];
        maxStr = now;
    }
}
```

###### 第二步：计算任意相邻两个之间间隔的最大公因子

​		下面我们把每个子串出现位置的头下标丢到 vector 里面，输出并计算任意相邻两个之间间隔的最大公因子 q。输出发现出现位置为 0 165 235 275 285，间隔的最大公因子 q = 5。下面是这部分代码：

```c++
/*
location : maxStr的出现位置
*/
vector <int> location;
for (int i = 0; i < L - 2; i++) {
    string now = "";
    for (int j = 0; j < cn; j++) now += s[i + j];
    if (now == maxStr)
        location.push_back(i);
}

/*
gcd_ij ： 任意相邻两个之间间隔的最大公因子
*/

int gcd_ij = 0;
for(int i = 1; i < location.size(); i++)
    gcd_ij = gcd(location[i] - location[i - 1], gcd_ij);
```

###### 第三步：重合指数法

​		原理里面提到过，我们这里可以猜测代换密钥的长度 m 是 q 的因子，所以我们枚举 q 的因子 i，对于每一个 i 中的每一组使用重合指数法计算 A ~ Z 的期望概率平方的累加和，然后看看这些组离 0.065 的绝对值的平均值在 i 等于多少时最小。我们记录最有答案下每一组的期望和对应的 m，发现m = 5时最优，这时每一组的平方期望和是0.0645161 0.0655738 0.0698043 0.0602856 0.0724484。下面是这一段的代码实现：

```c++
/*
m : 密钥长度
*/
int m = 0;
double min_abs = INF;
vector <double> m_p;
for (int i = 1; i <= gcd_ij; i++) {
    /*
    calc ： 密钥长度为i时的平均期望与0.65的差
    nowkey : 具体期望
    */
    if (gcd_ij % i != 0) continue;

    vector <int> cnt[i], tot;
    for (int j = 0; j < i; j++) cnt[j].resize(26);
    tot.resize(i);

    for (int j = 0; j < L / i * i; j++) {
        tot[j % i]++;
        cnt[j % i][s[j] - 'A']++;
    }

    vector <double> nowkey;
    double calc = 0;
    for (int j = 0; j < i; j++) {
        double now = 0;
        for (int k = 0; k < 25; k++)
            now += cnt[j][k] * (cnt[j][k] - 1);
        now /= tot[j] * (tot[j] - 1);
        calc += fabs(now - 0.065) / i;
        nowkey.push_back(now);
    }

    if (calc < min_abs) {
        min_abs = calc;
        m = i;
        m_p.clear();
        for (int j = 0; j < i; j++)
            m_p.push_back(nowkey[j]);
    }
}
```

##### 步骤二：破解密钥

​		接下来就是简单的单个破解移位密码了，对于每一组，我们都枚举移位的位数，看看对于移位多少最符合我们的统计规律。具体来说我们知道$\sum{p_i^2}=0.065$，当我们的数据符合统计规律时应该有$\frac{f_i}{n}=p_i$ ，所以我们只要替换一下，计算$\sum{\frac{p_if_i}{n}}$ 找到最接近 0.065 的就好了。我们最后发现这 5 组分别移位为 9 0 13 4 19，算出来的值为 0.0615806 0.067871 0.0655968 0.0596935 0.0724355，接近统计分析。因此得到密钥为：JANET。

```c++
/*
ansK : 最佳组位移（密钥）
ansP : 最佳位移的期望
*/

int nowN = L / m;
vector <int> num[m], tot;
for (int j = 0; j < m; j++)
    num[j].resize(26);
tot.resize(m);		
for (int j = 0; j < nowN * m; j++) {
    tot[j % m]++;
    num[j % m][s[j] - 'A']++;
}

vector <int> ansK;
vector <double> ansP;
for (int i = 0; i < m; i++) {
    ansP.push_back(INF);
    ansK.push_back(0);
    for (int k = 0; k < 26; k++) {
        double nowP = 0;
        for (int j = 0; j < 26; j++) nowP += num[i][(j + k) % 26] * p[j];
        nowP /= nowN;
        if (fabs(nowP - 0.065) < fabs(ansP[i] - 0.065)) {
            ansP[i] = nowP;
            ansK[i] = k;
        } 
    }
}
```

##### 步骤三：解密

​		最后我们用得到的密钥进行解密，得到答案。

```c++
cout << "译文 : ";
for (int i = 0; i < L; i++)
    cout << char((s[i] - 'A' - ansK[i % m] + 26) % 26 + 'A');
cout << endl;
```


### 三   实验结果

#### 实验结果

##### 编译参数和命令：

`g++ -std=c++11 -g -o Virginia Virginia.cpp`

`.\Virginia`

##### 运行结果：

```
出现次数最多的长度为3的子串 : CHR
出现了 5 次

CHR 出现的位置：0 165 235 275 285
任意相邻两个之间间隔的最大公因子：5

当表数为 1 时的平均期望与0.65的差 : 0.0200463
具体期望为 : 0.0449537
当表数为 5 时的平均期望与0.65的差 : 0.00360497
具体期望为 : 0.0645161 0.0655738 0.0698043 0.0602856 0.0724484
最优表数为 : 5

最佳 5 组移位为 : 9 0 13 4 19
其期望为 : 0.0615806 0.067871 0.0655968 0.0596935 0.0724355
密钥 : JANET
译文 : THEALMONDTREEWASINTENTATIVEBLOSSOMTHEDAYSWERELONGEROFTENENDINGWITHMAGNIFICENTEVENINGSOFCORRUGATEDPINKSKIESTHEHUNTINGSEASONWASOVERWITHHOUNDSANDGUNSPUTAWAYFORSIXMONTHSTHEVINEYARDSWEREBUSYAGAINASTHEWELLORGANIZEDFARMERSTREATEDTHEIRVINESANDTHEMORELACKADAISICALNEIGHBORSHURRIEDTODOTHEPRUNINGTHEYSHOULDHAVEDONEINNOVEMBER
```

#### 分析

​		前几个单词一看是比较有意义的，验证结果初步正确。稍加翻译，文章如下：

```
The almond tree was in tentative blossom. The days were longer often ending with magnificent evenings of corrugated pink skies. The hunting season was over with hounds and guns put away for six months. The vineyards were busy again as the well-organized farmers treated their vines and the more lack adaisical neighbors hurried to do the pruning. They should have done in november.
```

​		中文如下：

​		那棵杏树正在开花。白天更长，常常以波状粉红色天空的壮丽夜晚结束。狩猎季节结束了，猎犬和枪支被关了六个月。葡萄园又开始忙碌起来，因为组织良好的农场主在修剪葡萄，而缺乏远见的邻居们则忙着修剪。他们应该在11月份完成。


### 四   实验感想

​		本来想写点实验困难的，但由于课堂上讲述了完整的破译过程，而且算法确实也比较简单，所以没有遇到什么特别的困难。最大的困难可能也就是因为我调整过vscode的编码方式为`utf-8`，因此在这次使用`freopen`读取txt的时候一直读取不了（txt的名字是个中文），后来发现问题后将编码方式改写成`GBK`后就可以了。

​		这次的作业算是众多学科中比较有趣的一项作业，在破译的过程中充满了欢声笑语。在之前学习的组合数学与数论的基础上，结合统计学的知识，这次是实验让我对古典密码的破译过程有了更加深刻的认识。在完成作业的同时，也巩固了课堂所学的知识。同时也感悟到：密码学的工作需要结合很多学科的知识，是一门非常有趣的学科，值得钻研。

​		总的来说，还是获益匪浅的。
