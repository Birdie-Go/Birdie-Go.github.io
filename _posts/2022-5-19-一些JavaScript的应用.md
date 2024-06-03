---
layout:     post
title:      一些JavaScript的应用
subtitle:   专业技术综合实践
date:       2022/5/10
author:     Birdie
header-img: img/post_header_sr.jpg
catalog: true
tags:
    - JavaScript
    - HTML
    - CSS
---

## 关于JavaScript的应用

记录了一些作业的过程......比较随意，因为不是实验报告，也就是笔记而已，格式就不要求自己了吧。



### 验证密码

输入两个密码，如果不相同则会弹出信息。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image002.jpg" alt="img" style="zoom:70%;" />

随意设计的界面，毕竟我也没有加css进去，有点丑陋。

如果两次输入的密码不相同，则会有：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image002-16525184499451.jpg" alt="img" style="zoom:50%;" />

相同的话就没有处理了。

![image-20220514165438094]({{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514165438094.png)

markdown里面放html代码会和网页生成的html混在一起，所以还是放图片吧。

主要还是使用document中的getElementById语句，获取对应id的元素，并从value中取出对象对应的内容。

alert是弹出的警告框。



### 幻灯片

可能结合图片会更清楚是什么。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514165810364.png" alt="image-20220514165810364" style="zoom:80%;" />

其实就是那种滚动的图片。当鼠标在图片上方时，显示左右箭头，并停止动画。当鼠标离开画面时，隐藏左右箭头，重新启动动画。动画是会隔2s中自动向左滑动到下一张图片。可惜markdown不能放视频，其实可以gif，不过我比较懒。每次点击左右箭头将移动一副图像。右下角的三个点每个点对应一幅图像，当前图像的变化时所对应的点也会变化。点击它们时会显示对应的图像。

#### css

css的部分就不细说了，毕竟之前练了一些。

(1) 图片采用三个img，并用一个div包含，样式white-space要加上nowrap。这样使得三个图片并列在一行上，同时设置div的高度和宽度为一行图片的高度和宽度。

(2) 外层再用一个div，设置宽度为一个图片的宽度，并设置overflow为hidden。

(3) 上面的两个div均为relative。

(4) 每个点和箭头采用div和背景图实现，三个点再采用一个div封装，该div和两个箭头的div均采用绝对定位。

就这样吧，意思意思一下。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514171033780.png" alt="image-20220514171033780" style="zoom:50%;" /><img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514171053574.png" alt="image-20220514171053574" style="zoom:50%;" />

#### JavaScript

先来学习如何获取多个元素。在document中有querySelector和querySelectorAll可以获取元素。前者取得与CSS选择器匹配的第一个节点，后者取得与CSS选择器匹配的所有元素的集合，都只限于html5。

幻灯片的第一步是**动画**。动画用setTimeout的定时器功能，该方法用于在指定的毫秒数后调用函数或计算表达式。首先我们设置一个全局的id表示当前的图片的id，flag表示鼠标是否正在画面上方，而pos表示这三张图片的x轴的位置。动画设计为，0.1s内完成一张图片的切换，然后停留2s。具体来说，一张图片的宽度是773ps，也就是一帧移动7.73px。用位置来控制定时任务即可。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514180734638.png" alt="image-20220514180734638" style="zoom:50%;" />

然后是**箭头显示和隐身的部分**，这其实就是控制箭头这个元素的display，none表示不显示，而block表示显示。同时需要控制一个flag标记，用于控制动画是否继续进行。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514180419559.png" alt="image-20220514180419559" style="zoom:50%;" />

箭头切换图片和点切换图片的部分，和动画的部分有一些不一样。动画是缓慢移动的，而点击箭头和点，图片应该瞬间变换。那这更好办了，直接写一个通项公式就可以了。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220514180430412.png" alt="image-20220514180430412" style="zoom:50%;" />

#### 注意要点

有一些要点是要注意的，比如**scrpit放的位置**：

1. head标签内：外部引入的js文件，或者不涉及页面元素操作的js代码块。

2. body标签内：放哪都行，只要保证放在操作元素下面就行，但是如果需要自执行js就需要放在这里面。一般推荐放在最后面。

一开始我就是放在了head里面，发现动画不动（我记得老师上课没有讲吧，不排除我走神了，但是我记得我好好听课了的）。

**onmousemove和appearmouse的位置**，我一开始设置是在图片的div元素上的，也就是id是images的位置。后来发现，我的鼠标移动到箭头上方的时候，箭头会不停地闪烁。确实，因为箭头和点都不属于这个图片元素。所以还是得把onmousemove和appearmouse设置到整个元素上面，也就是main元素上。

**setTimeout的使用**也有一点点考究，用于在指定的毫秒数后调用函数或计算表达式，意思是他只会执行一次。我就说我一开始讲setTimeout放在主函数体中，动画怎么不动。其实应该放在move的函数体中，回调调用。

还有一些很愚蠢的错误就不说了。



### 菜单

先前已经用css设计过一个菜单，现在尝试用JavaScript也做一个。

我们的目标是这个样子的：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image002-16526036426611.jpg" alt="img" style="zoom:50%;" />

#### html body

通过 li 元素写好html后，在未设置css前，长这个样子：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220515163436004.png" alt="image-20220515163436004" style="zoom: 33%;" />

body部分大概长这个样子（url什么的就不要管了）：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220515163514454.png" alt="image-20220515163514454" style="zoom:50%;" />

#### css

首先第一步，是设置css（怎么也得先排版好吧）。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517171050291.png" alt="image-20220517171050291" style="zoom: 50%;" />

这样就通过鼠标hover的方式实现了这个菜单。

#### JavaScript

那么现在将hover方式改写成JavaScript。大概的思路就是取消css的hover功能，并用mouseover和mouseout进行实现。

![image-20220517173008029]({{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517173008029.png)

但是我不知道为什么改写后子菜单的位置会变化，转而要改一下left和top。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517172938652.png" alt="image-20220517172938652" style="zoom:50%;" />

#### 注意要点

设置了好久背景图片都没有显示，background-image不生效，但是background却是生效的。很奇怪，设置了display、background-position、width、height等能调整的都调整了，仍然不奏效。最后想到一个办法，debug的时候直接把span标签改成了img标签，马上发现病痛所在。我发现它显示的图片是找不到该图片。但是我很清楚图片的路径是没有问题的。那问题在哪，我往上翻，发现了一个很致命的标签。老师设置了base标签，将所有的相对路径改成了绝对路径。我注释掉这句话，终于看见那张图片了。在网上查了好久如何破解base，毫无办法。于是，在不改动html的情况下，我将这个base标签放在了style下方，这样就影响不到我的图片了。



### DOM

DOM用正文为空的dom.html，通过javascript的DOM操作，实现以下html网页(jys.html)： 

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517195504464.png" alt="image-20220517195504464" style="zoom: 80%;" />

首先第一步，把样子设计好。就直接用html和css开写了：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517194057479.png" alt="image-20220517194057479" style="zoom: 50%;" />

然后就按照这个样子，将html元素用js的形式创建。

- 节点的创建采用document.createElement()、document.createTextNode()和e.cloneNode()实现；

- 加入元素采用e.appendChild()实现。

- 可以用document.body.appendChild(e);把元素加入当前网页。

 步骤为：

1. 先设计第一行，除了第一个上标，第二上标采用克隆第一个上标实现。
2. 上标中的属性采用e.style.verticalAlign="super"和e.style.fontSize="12px"实现
3. 第二行采用克隆第一行实现，利用childNodes和nodeValue修改文本节点的内容。

注释掉html和css，加入如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220517195446318.png" alt="image-20220517195446318" style="zoom:50%;" />



### Ajax-get技术

#### Ajax简介

传统的网页如果需要更新内容，必需重载整个网页。使用AJAX (Asynchronous  Javascript And XML) 技术除了可以利用从 Web 服务器上获取的信息局部更新网页， 还可以实现在保持当前网页的情况下提交数据给Web服务器。 

2005年2月，Adaptive Path公司的Jesse James Garrett在文章“Ajax: A New  Approach to Web Applications”中最早提出这个概念。这篇文章认为将XHTML、CSS、 JavaScript、DOM和XMLHttpRequest混合使用来开发Web应用将会成为一种新的趋势。 事实上，在Ajax这个概念出现之前就已经有了丰富的Ajax应用，例如，Google Maps 和Google Suggest就是应用了XMLHttpRequest异步从服务器端来获取数据，实现了 客户端无刷新的效果。

所有现代浏览器均支持 XMLHttpRequest 对象（IE5 和 IE6 使用 ActiveXObject）。

#### 实验要求

有一个这样的记录大概长这个样子：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518202801020.png" alt="image-20220518202801020" style="zoom:67%;" />

需要实现一个ajaxGet.html，初始画面是这样的：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518202705259.png" alt="image-20220518202705259" style="zoom:67%;" />

输入记录中存在的id：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image006.jpg" alt="img" style="zoom:50%;" />

离开id输入域后利用ajax-get技术调用get.jsp，从stu表中取得num和name并显示出来，返回的格式见下图：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518204405626.png" alt="image-20220518204405626" style="zoom:50%;" />

关闭弹出框后：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image010.jpg" alt="img" style="zoom:50%;" />

#### 实现

这部分技术需要一些http协议的知识。

初始画面的设计如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518202629103.png" alt="image-20220518202629103" style="zoom:50%;" />

加入JavaScript后如下：

![image-20220518204740887]({{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518204740887.png)



### Ajax-post技术

#### 实验要求

初始界面如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518205509795.png" alt="image-20220518205509795" style="zoom:67%;" />

先输入id，当输入新的姓名并离开时用ajax技术调用post.jsp（参数为id和name）修改具有该id记录的姓名的值。

#### 实现

![image-20220518212755272]({{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518212755272.png)

#### 实验结果

原id：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518212656587.png" alt="image-20220518212656587" style="zoom:67%;" />

填写如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518212231014.png" alt="image-20220518212231014" style="zoom:67%;" />

离开文本框后：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518212552765.png" alt="image-20220518212552765" style="zoom: 67%;" />

修改成功！

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518212734965.png" alt="image-20220518212734965" style="zoom:67%;" />



### 富文本编辑

input元素或者textarea元素只能输入文字，富文本编辑可以在输入框中直接设置文 本颜色、字体大小、显示图片等。 

有两种实现富文本编辑的方法，一种是通过iframe实现，另一种是通过div实现。 它们都可以对当前选中的内容通过执行命令设置样式。

采用iframe实现的过程如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518213810792.png" alt="image-20220518213810792" style="zoom:67%;" />

但实验是用div实现的。

有一个关键方法：execCommand方法是执行一个对当前文档，当前选择或者给出范围的命令。处理Html数据时常用如下格式：document.execCommand(sCommand[,交互方式, 动态参数])。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518215306912.png" alt="image-20220518215306912" style="zoom:67%;" />

具体的交互方式和动态参数可以查看文档。



完成后如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518215631064.png" alt="image-20220518215631064" style="zoom:50%;" />

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518215614792.png" alt="image-20220518215614792" style="zoom: 50%;" />

输入：A quick brown fox jumps over the lazy dog 并逐个单词试一下功能。

（注意，size是内置函数，不要起这个函数名）

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518220126605.png" alt="image-20220518220126605" style="zoom:50%;" />

按撤销后fox的链接消失，按源码后显示

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518220405514.png" alt="image-20220518220405514" style="zoom:50%;" />



### 消息弹框

编写一个消息框弹出程序，当无输入值时显示提示信息，否则显示输入值。要求消息框定位在浏览器客户区中央，并且整个窗口(包括被遮蔽的网页部分)变灰且禁止操作，直到点击信息框的OK按钮才变正常。消息框要求有标题栏和关闭按钮。

#### 网页禁止操作

我们设计了这个网页教学如何网页变灰且禁止操作。

点击变暗后，页面确实无法操作了，滚下去的网页也不行。

![image-20220518222355316]({{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518222355316.png)

代码如下，其实很显然是设置了一个页面大小的灰色的内联块元素，置于顶层使得无法操作网页的元素。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518221750143.png" alt="image-20220518221750143" style="zoom:50%;" />

#### 中间定位

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518222556847.png" alt="image-20220518222556847" style="zoom:67%;" />

点击后

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518222614961.png" alt="image-20220518222614961" style="zoom:50%;" />

其实就是使用了JavaScript的一些内置变量，clientWidth和clientHeight存储浏览器页面的宽度和高度。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220518222759158.png" alt="image-20220518222759158" style="zoom:50%;" />

#### 实现

那么思路就显现了：

1. 默认弹窗和灰色禁止块是隐藏的；
2. 点击按钮后，执行网页禁止操作和显示弹窗；
3. 网页禁止操作即将一个div模块设置为block；
4. 设计显示弹窗并显示在中间；
5. 最后实现提交动作。

其实代码写的很清晰，先不看css的部分，html的主题部分如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519004409959.png" alt="image-20220519004409959" style="zoom:50%;" />

shadow是阴影部分，而带有msg的是消息框。最后那部分是原始的画面。

函数的主体为：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519004508325.png" alt="image-20220519004508325" style="zoom:50%;" />

具体各模块为：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519004528468.png" alt="image-20220519004528468" style="zoom:50%;" />

个人感觉还挺清晰的。

css的部分如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519004257840.png" alt="image-20220519004257840" style="zoom:50%;" /><img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519004556892.png" alt="image-20220519004556892" style="zoom:50%;" />



#### 实验结果

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image002-16528921144042.jpg" alt="img" style="zoom:67%;" />

点击check后：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image004-16528921144043.jpg" alt="img" style="zoom:67%;" />

输入内容后，在此点击：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image006-16528921144044.jpg" alt="img" style="zoom:67%;" />

窗口变小了依旧位于中间：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/clip_image008-16528921144045.jpg" alt="img" style="zoom:67%;" />



### 动态图片加载

为了节约流量，让网页中的图像只有出现在浏览器客户区时才显示。

方法是用自定义属性xSrc保存图像的URL，在网页滚动时对网页中所有图像(image元素)进行判断，看看它们的上边沿或下边沿是否出现在浏览器客户区，如果是，则把xSrc的值拷贝到属性Src中（用getAttribute 和setAttribute方法），此时浏览器才去下载并显示图像。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519173433852.png" alt="image-20220519173433852" style="zoom: 50%;" />

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519173443778.png" alt="image-20220519173443778" style="zoom:50%;" />

示例如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519173537626.png" alt="image-20220519173537626" style="zoom:50%;" />

#### setAttribute()

使用元素的 setAttribute() 方法可以设置元素的属性值。用法如下：

```javascript
setAttribute(name, value)
```

参数 name 和 value 分别表示属性名称和属性值。属性名和属性值必须以字符串的形式进行传递。如果元素中存在指定的属性，它的值将被刷新；如果不存在，则 setAttribute() 方法将为元素创建该属性并赋值。



### 计时器

设计一个多计时器(秒表)的Web页，每次点击“增加计时器”按钮会增加一个计时器。

每个计时器从0开始计时：

- 如果按了暂停键，则停止计时，此时只有按启动键才继续计时，按其他键无效；
- 如果按了停止键，则停止计时，此时只有按启动键才从0开始计时，按其他键无效；

#### html和css

整体框架如下：

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519174003725.png" alt="image-20220519174003725" style="zoom:50%;" />



#### addEventListener

[HTML DOM addEventListener() 方法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/jsref/met-element-addeventlistener.html)

addEventListener会先调用一次绑定的函数，因此函数应该写成回调函数的形式。

#### JavaScript

先设置计时器函数的原型。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519231622582.png" alt="image-20220519231622582" style="zoom:50%;" />

然后再设置计时器的功能。个人觉得注释写的还挺详细的。

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519231700120.png" alt="image-20220519231700120" style="zoom:50%;" />

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519231735432.png" alt="image-20220519231735432" style="zoom:50%;" />

#### 实验结果

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519231913142.png" alt="image-20220519231913142" style="zoom: 33%;" />

<img src="{{site.url}}/img/2022-5-19-一些JavaScript的应用/image-20220519231936346.png" alt="image-20220519231936346" style="zoom: 33%;" />

#### 感悟

做完这个实验我才领悟到：因为js没有类的概念，是用函数去模拟类的。原型就类似基类，闭包就类似类中的成员函数。

怎么说呢，实验过后对理论上学习到的知识更加深刻了。
