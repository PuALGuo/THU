# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt
import torch
from net import TorchNet, TFNet
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

def test_TF():

    model=TFNet()

    tf.saved_model.save(obj=model, export_dir="./out/tensorflow/model")

    # frozen_model
    '''
    TVM不支持直接读取tf模型，需要将模型转化成frozen_model
    有点类似于动态模型转化成静态模型的感觉
    代码是网上抄的，所以我不知道原理
    '''
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec((1,3,64,64), 'float32'))
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    
    layers = [op.name for op in frozen_func.graph.get_operations()]
    
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./out/tensorflow/frozen_models",
                      name="test.pb",
                      as_text=False)

    # graph_def
    '''
    根据frozen_model转化成可以被分析的graph_def形式，api来源于tensorflow1的GraphDef()
    没有找到tensorflow2的支持信息，官方也没有给出信息
    '''
    with tf_compat_v1.gfile.GFile('./out/tensorflow/frozen_models/test.pb', "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # with tf_compat_v1.Session() as sess:
        #     graph_def = tf_testing.AddShapesToGraphDef(sess, 'Identity')


    # input_node
    '''
    输入节点标记
    '''
    input_node_name = [node.name for node in graph_def.node if len(node.input)==0 and node.op not in ('Const')]
    input_node_name = input_node_name[0]
    shape_dict = {input_node_name : (1,3,64,64)}

    # output_node
    '''
    寻找最终节点（无后继），默认是最后一个节点
    '''
    graph_dict = dict()
    for node in graph_def.node:
        graph_dict[node.name] = node.input
    for name_src in graph_dict:
        found = False
        for name_dst in graph_dict:
            if name_src == name_dst: continue
            if name_src in graph_dict[name_dst]:
                found = True
                break
        if not found:
            output_node_name = name_src
            break

    # Addshape
    '''
    大概就是在output节点上打个标记，表示最终输出，默认情况下是最后一个节点
    '''
    with tf_compat_v1.Session(graph=graph) as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, output_node_name)

    layout = 'NCHW'
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict, outputs=[output_node_name])
    f = open('./out/tensorflow.txt','w')
    print(mod['main'],file = f) ## 需要声明是main，现在的翻译转化会产生很多没有用的函数信息
    f.close()

def test_Torch():

    model = TorchNet()
    input_shape = [1, 3, 64, 64]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval() ## torch模型需要用jit编译，pytorch自己提供的api

    x = np.random.rand(1,3,64,64)

    input_name = "input0"
    shape_list = [(input_name, x.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    f = open('./out/torch.txt','w')
    print(mod['main'],file = f) ## 需要声明是main，现在的翻译转化会产生很多没有用的函数信息
    f.close()

if __name__ == "__main__":
    
    test_TF()

    test_Torch()

    print('everything is done')