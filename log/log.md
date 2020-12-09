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
2. 