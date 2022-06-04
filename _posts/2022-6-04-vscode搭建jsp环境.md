---
layout:     post
title:      vscode搭建jsp环境
subtitle:   含泪总结七个小时的教训
date:       2022/6/04
author:     Birdie
header-img: img/post_header.jpg
catalog: true
tags:
    - JSP
    - tomcat
---

前言：Eclipse 确实一下子就可以搭建完了，但是我的确不喜欢用 Eclipse。

在搭建 jsp 环境的时候，浪费了很多时间。网络上的资料不是很全，而且确实环境搭建比较复杂。这里讲一下我个人搭建过程中的一些细节。

推荐阅读：[使用VSCode进行Java+Maven+Tomcat环境搭建](https://tieba.baidu.com/p/6546145317)

### 1 准备

我个人这么理解 jsp，可以理解为 java 版的 php。

需要安装的东西有：

- JDK：[Java Downloads | Oracle](https://www.oracle.com/java/technologies/downloads/)

  我个人装的是 JDK17，都没有太大的问题，选择一个稳定的版本安装即可。jsp 其实就是 java，当然要装 jdk。

- 一个 IDE：我用的 vscode，我曾经尝试过用 Eclipse，但是没用过，用不顺，有学习成本。虽然 Eclipse 搭建环境挺方便的，但我个人还是喜欢 vscode。当然也可以用 vim，但是 IDE 的插件可以简化搭建的过程。

  如果是 Eclipse 的话，需要安装 Eclipse IDE for Enterprise Java and Web Developers。而且网上有一些教程没有告诉你，其实还需要一个叫做 Web, XML, Java EE and OSGi Enterprise Development 的工具。可以去官网下，也可以在 Help 里面下：[解决Eclipse新建中没有Dynamic Web Project的问题](https://blog.csdn.net/qq_29774003/article/details/108023831)。

- ApacheMaven：[Maven – Download Apache Maven](https://maven.apache.org/download.cgi)

  一个项目管理工具。

- ApacheTomcat：[https://tomcat.apache.org/download-90.cgi](http://jump.bdimg.com/safecheck/index?url=rN3wPs8te/rXWOXKb4/RlWqSH7g/bko4FX8KEwN3B7cBMcC3K0MHAZ0MoKoKSpY7SqPYof0Oc1klrr33yaWzULK9GagrK0KPCUxI03BAgb3yP8OY5FVEZTldKfFr8Ighxxk/J3okjsDE6imU4OLCJ1ZmEBVHjGgoMDxm7iZ2BjQ=)

  一个 Web 应用服务器，jsp 项目就是部署在这个服务器上。

### 2 环境变量

**注意：安装路径必须是纯英文且不带空格的。**

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604212207457.png" alt="image-20220604212207457" style="zoom: 33%;" />

1. 将 JDK 中的 bin 目录放进环境变量，在 cmd 中执行 `java -version` 测试是否成功。重启 cmd 是要的，反正还不成功就重启吧。

2. 将 Maven 中的 bin 目录放入环境变量，在 cmd 中执行 `mvn -version` 测试是否成功。

   Maven 的服务器设在境外，可以设置阿里云的镜像。打开 conf 中的 `settings.xml` 文件，在 `<mirrors>` 中加入：

   ```xml
   <mirror>
   <id>alimaven</id>
   <name>aliyun maven</name>
   <mirrorOf>central</mirrorOf>
   <url>http://maven.aliyun.com/nexus/content/repositories/central/</url>
   </mirror>
   ```

   <img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604205943336.png" alt="image-20220604205943336" style="zoom:50%;" />

3. 将 Tomcat 中的 bin 目录放入环境变量。在 bin 目录下，打开 PowerShell，有的说执行 `service.bat install` 完成服务的安装。反正这一步我是没有成功，因为我没有`service.bat`，如果有 `startup.bat`，可以直接双击运行。打开浏览器，在网址栏输入 `localhost:8080`，如果有 tomcat 的画面，那就是成功了。端口号是可以改的，如果出现的端口冲突，那就在 conf 目录下的 `server.xml` 里面搜索 8080，然后替换成别的不冲突的端口， 比如 8000。

### 2.5 vscode 要干的活

如果用 vscode，有一些插件可以装上：

- Java Server Pages (JSP)：这个是 jsp 的语法高亮，但是它已经不维护了，勉强可用，比没有强。
- Maven for Java：这个是自动搭建框架的。
- Tomcatfor Java：在服务器上运行的插件。
- JavaExtension Pack：java 的插件。

安装完后可以重启一下 vscode，vscode 好像没有重启的按钮，关掉再开吧。

然后打开一个你将要用来写 jsp 的文件夹，用 vscode 打开这个文件夹。如果安装了 Tomcatfor Java 的插件，可以看到左下角有一个 TOMCAT SERVERS，点击那个加号，然后目录选择 apache-tomcat 的根目录，比如 apache-tomcat-9.0.31。

### 3 创建工程

**首先要确认，后台没有运行刚刚打开的 `startup.bat`，确保要关掉，不放心就运行 `shutdown.bat`。**

左下角有一个 MAVEN 栏，点 + 号。

<img src="http://tiebapic.baidu.com/forum/w%3D580/sign=59ec0443dffcc3ceb4c0c93ba244d6b7/eb6e8e12b07eca807bc4b186862397dda044830a.jpg?tbpicau=2022-06-06-05_fedca3d67265809f35e55315476c6548" alt="img" style="zoom: 67%;" />

然后顶端会弹出框框，输入 webapp，选 maven-archetype-webapp。

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604212528949.png" alt="image-20220604212528949" style="zoom:50%;" />

然后选 1.4，我的意思是选哪个不重要。

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604212622355.png" alt="image-20220604212622355" style="zoom:50%;" />

然后会让你写一些东西，自己看着办吧，其实就是项目的名称、文件夹名之类的。当然工程上要按照规范来。

然后终端也会让你输入东西，你看着它让你写什么你就写什么就好，不影响环境的搭建。

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604212917542.png" alt="image-20220604212917542" style="zoom:50%;" />

最终文件结构长这样

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604213017946.png" alt="image-20220604213017946" style="zoom:50%;" />

然后就可以在 webapp 文件夹内工作了。WEB-INF 中可以建一个 lib 文件夹，用来放一些需要用的包。

编译的话就选中 webapp，然后右键

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604213236521.png" alt="image-20220604213236521" style="zoom: 33%;" />

终端可以看到编译的结果

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604213505992.png" alt="image-20220604213505992" style="zoom:50%;" />

左边可以看到 tomcat 服务器编程绿色，表示正在运行。右键 apache-tomcat 选择在浏览器中打开

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604213605760.png" alt="image-20220604213605760" style="zoom: 33%;" />

打开是这个画面

<img src="{{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604213654444.png" alt="image-20220604213654444" style="zoom: 33%;" />

文件夹中就是你刚刚新建的工程，点击就可以显示 jsp 生成的网页。



### 注意

#### 内存泄漏报错

如图所示

![image-20220604110242572]({{site.url}}/img/2022-6-04-vscode搭建jsp环境/image-20220604110242572.png)

出现这个错误的原因就是，服务器在后台已经启动了。要么就是 `startup.bat` 在后台工作，要么就是你上一次打开服务器太久没响应，重启一下即可。或者到 tomcat 的 bin 目录下，运行 `shutdown.bat` 关掉后台。

#### 关于executeQuery

我没看到相关的资料，但是根据我的实验，`executeQuery` 执行后是会关闭上一次的结果集的。举个例子：

```java
String sql_1 = "select * from user;";
ResultSet rs_1 = stmt.executeQuery(sql_1);
while (rs_1.next()) {
    String sql_2 = "select * from blog;";
	ResultSet rs_2 = stmt.executeQuery(sql_2);
    // 这个地方 rs_1 就被关闭了
}
```

因此建议不要在下一次 `executeQuery` 后仍对上一个查询的结果集操作。

#### 关于 jsp 输入变量

<%=变量%> 可以输出一个变量。

#### jsp 与 mysql

可以看这篇：[JSP中使用数据库(mysql)](https://blog.csdn.net/qq_42907061/article/details/117716103)

如果处理完想返回上一页，可以

```javascript
<script>
    window.history.go(-1);
</script>
```

还有这篇：[window.history.go(-1)返回上页的同时刷新"上页"技术](https://blog.csdn.net/educast/article/details/2895006)