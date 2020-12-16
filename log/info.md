**额外写一篇专门给内容的，非时间顺序的**

# TVM的前端转化

## MXNet

### 框架支持
*框架的api支持*
  
### TVM支持
*TVM内部的api支持*
+ relay.frontend.from_mxnet
  + 对输入进行check，Only Hybrid Blocks are supported now
  + 解析arg_params和aux_params，我没见过这个例子，有了再来补充
  + mxnet需要进行一次前向传播才会建立完整模型（如果导出还需要Hybridize()）
+ _from_mxnet_impl
  + json.loads(symbol.tojson())读取graph信息（mxnet的api）
  + 检查是否有unsupported的op
  + 遍历graph的node，建立对应op
    + 对于null类型的op（数据），建立_expr.var
    + 对于真实的op类型，参考_convert_map建立op 
    + ["head"]信息应该是终点信息（大概吧，根据其建立output
    + _function.Function(analysis.free_vars(outputs), outputs)建立完整的中间表示
    函数会自动遍历outputs节点，自动推理建立完整结构，free_vars没怎么了解，大概是终点是浮空的，当作外部接口？
  + _identity_list: ops in the identity set must be attribute free
  + _convert_map: op映射表
  + 还有我没有用过的，我也不知道干嘛的表
    + _control_flow_ops = ['_cond', '_foreach', '_while_loop']
    + _qnn_subgraph_ops = ['_sg_mkldnn_conv', '_sg_mkldnn_fully_connected']
    + _subgraph_ops = _control_flow_ops + _qnn_subgraph_ops
    + _params_ops = ['_contrib_quantized_ring_buffer']

# 额外内容

## TVM和THU的IR对比
*相比于TVM的IR设计略显简单粗糙*
+ 缺少数据节点定义，用了简单的'input'节点但没有明确定义
+ TVM中nodes/shape/dtype分开定义
  + 好处是减少了重复定义，THU这个输入输出都需要定义
  + 会存在同一个数据给两个节点供数但dtype等信息不同吗？
+ TVM给出storage Node定义
  + 这个跟需求有关系，比较难定义
+ TVM给出了整体图的arg_nodes，node_row_ptr之类的定义
  + 大概对重建图有帮助
  + 但默认是顺序读取数据图的情况下差别不大
  + 乱序数据图中TVM更方便？但THU版本也不是不能建立
  + 个人感觉如此，对这些内容的定义不是很明确

## 更加额外的内容
*还没有被整理或者不知道怎么整理的内容*
+ tensorflow中的bias被识别成了Add，而torch的bias被识别成了bias_add
+ upsample被转换成了image.resize,但是信息什么的还在
+ mod文件需要声明是main，现在的翻译转化会产生很多没有用的函数信息
+ 关于tf
  + frozen_model
    + TVM不支持直接读取tf模型，需要将模型转化成frozen_model
    + 有点类似于动态模型转化成静态模型的感觉
    + 代码是网上抄的，所以我不知道有原理
  + graph_def
    + 根据frozen_model转化成可以被分析的graph_def形式，api来源于tensorflow1的GraphDef()
    + 没有找到tensorflow2的支持信息，官方也没有给出信息
  + input_node & output_node
    + tf的转化需要标记输入输出节点
+ 关于torch
  + torch模型需要用jit编译，pytorch自己提供的api