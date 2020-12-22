~~markdown崩掉了，大概是vscode版本更新的缘故，两者的快捷键冲突了~~ 

~~给大家表演一个从头再来~~

---
# 20201207
1. upsample被转换成了image.resize,但是信息什么的还在
2. tensorflow中的bias被识别成了Add，而torch的bias被识别成了bias_add
3. mod文件需要声明是main，现在的翻译转化会产生很多没有用的函数信息
4. 关于tf
  + frozen_model
    + TVM不支持直接读取tf模型，需要将模型转化成frozen_model
    + 有点类似于动态模型转化成静态模型的感觉
    + 代码是网上抄的，所以我不知道有原理
  + graph_def
    + 根据frozen_model转化成可以被分析的graph_def形式，api来源于tensorflow1的GraphDef()
    + 没有找到tensorflow2的支持信息，官方也没有给出信息
  + input_node & output_node
    + tf的转化需要标记输入输出节点
5. 关于torch
  + torch模型需要用jit编译，pytorch自己提供的api

~~*P.S. \<br/>\</br> 这是声明换行*~~

---
# 20201209
1. to_json载入的时候卡了，我还以为这几天活我白干了

---
??? 啥玩意，为啥东西没了?

---
# 20201217
## 今日目标
+ ~~填补丢失的信息~~
+ 哭一会
+ 整理一下TVM的前端转化工作
+ exp()函数
+ 毕设相关内容
## 今日进程
+ 填补丢失的信息
  + gem5不支持python语言编译出的结果
  + 存在系统调用fchmod，虽然我也不能理解为什么有这个系统调用，但gem5的se模式不支持
  + 同理，java等非C语言也有类似的问题
  + 或许可以试试gem5的fs模式，但也没啥时间了
+ THU的前端设计部分
  + 幸好我额外写了一个info.md，应该大部分信息都还在
+ 哭了，tmd都怪sourcetree

---
# 20201221
## 今日目标
+ ~~exp()函数拟合~~
+ 毕设相关
+ ~~TVM前端转化~~

## 今日进程
+ exp()函数拟合
  + 选择用32项的泰勒展开进行拟合
  + 在0.5 ~ -0.5 区间差距甚大，用线性拟合最低到0.0002精度
+ TVM前端转化分析
  + 摸完了
