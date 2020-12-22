**额外写一篇专门给内容的，非时间顺序的**

# TVM的前端转化

## MXNet

### 框架支持
*框架的api支持*

1. Sequential
Sequential构建的是动态图，即命令式编程形式，类似于pytorch的那种形式，可以用命令行交互的进行调试而不用先构建完整的图，同时可以输入变量而不是用符号（指的是tf那种placeholder的形式）进行调试

2. HybridSequential
实际上应该是HybridSequential和HybridBlock，这是TVM支持的模式。其可以进行命令式和符号式的混合，可以在动态图和静态图之间转换，使用者可以先用imperatvie的方式写网络，debug，最后跑通网络之后，如果网络是一个静态图结构，就可以用net.hybridize()的方式将其转换成静态图，众所周知静态图的运算会比动态图快，所以这是Gluon比PyTorch更好的地方。

3. 一点题外话
从这里就可以很明显的看出，TVM实际上支持的是静态图的转化，因为网络的识别和提取只需要通过访问静态图得到而不用去识别模型架构本身，从而绕过了框架的差异性进行通用化表达。

### TVM支持
*TVM内部的api支持*
+ shape_dict
  + shape_dict = {"data": x.shape}
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

## Pytorch

### 框架支持

1. 动态图->静态图
Pytorch自身实现的是动态图设计，因此需要将动态图转化为静态图输入给TVM，使用的是pytorch自带的torch.jit.trace(model, input_data).eval()

2. aten里的函数对应
之前跑的时候有一些aten实现的函数对应失败，但没有做记录忘记有哪些了，有机会了补充

### TVM支持
+ shape_dict
  + [(input_name, img.shape)]
+ relay.frontend.from_pytorch
  + Prelude()
    + pytorch这个工具看起来像是用词法/语法分析树啊...tree_adt都出来了
    + Parses the Prelude from Relay's text format into a module
  + _get_convert_map
    + 看上去是像是针对quantized_tensor所以包装了一下
    + 反正最终的功能就是实现一个转换的map
  + script_module.graph
    + torch自带api，类似地是为了生成graph，当然跟mxnet的格式不太一样
    + torch内部打包的layer没有被解析
      + 形如 __torch__.torch.nn.modules.module.___torch_mangle_16.Module = prim::GetAttr[name="layer1"](%self.1)
    + 不像其他的，他是逆向建树，从return开始的（我看这个数据结构猜的）
      + 从顺序上看，Module(op) - CallMethod - Data
  + _run_jit_passes
    + 续上，解析图形成更详细的图表示
    + 他把一些参数，例如padding之类的也看成节点写在图中了
  + get_all_op_names
    + findAllNodes可以获取graph中的节点信息（pytorch的api）                   
  + _report_missing_conversion
  + _get_relay_input_vars
    + 实际上返回的就是在IR中打印的ty参数，(shape, dtype)
    + 他为什么函数是input？？？
    + 返回的标记为output
  + qnn
    + qnn_torch.add_input_quant_params_to_op_inputs(graph)
      qnn_torch.add_quant_params_to_outputs(outputs,
                                            packed_param_map,
                                            weight_quant_params)
      qnn_torch.add_quant_params(tvm_params, weight_quant_params)
    + 针对pytorch的量化写了专门的代码
    + 量化的操作用的是pytorch的api
  + convert_operators
    + 将graph转化为对应的节点表示/IR表示
    + 就转化，1对1转换，以及特化的wrap描述

## Tensorflow2
*就比其他要复杂的多*

### 框架支持

1. 模型转换
TVM不支持直接读取tf模型，需要将模型转化成frozen_model。有点类似于动态模型转化成静态模型的感觉。但本质上不是这个意思，具体原理也不是很懂

2. frozen_func.graph.as_graph_def()
解析模型并转化成图定义格式

3. tf.io.write_graph()
tf的官方api用以导出graph，.pb格式

4. 另一端的载入
tf_compat_v1.gfile.GFile()用来读取pb文件，tf_compat_v1.GraphDef()进行初始化，graph_def.ParseFromString(f.read())进行载入，tf.import_graph_def(graph_def, name="")成图， tf_testing.ProcessGraphDefParam(graph_def)检查和规范化graph_def

5. 标记输出
tf_testing.AddShapesToGraphDef(sess, output_node_name)标记输出节点（这功能是我猜测的没有细察

### TVM支持
+ 节点标记
  + 利用tf的api获取每个节点的名称，确定输入节点和输出节点
+ shape_dict
  + input_node_name = input_node_name[0]
  + shape_dict = {input_node_name : (1,3,64,64)}
+ relay.frontend.from_tensorflow
  + 他这个代码风格明显跟其他的不同啊
  + 就本质上还是翻译过程
  + 累了，下次有机会再写
  
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