# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.apache.org/versions/master/install/index.html
"""
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
model = TFNet()
# input_shape = [1, 3, 64, 64]
# input_data = torch.randn(input_shape)
# scripted_model = torch.jit.trace(model, input_data).eval()

x = np.random.rand(1,3,64,64)
print("x", x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon

## mxnet的语法
# shape_dict = {"data": x.shape}
# mod, params = relay.frontend.from_pytorch(model, shape_dict)

## torch
# input_name = "input0"
# shape_list = [(input_name, x.shape)]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

## TF
shape_dict = {"input_1": x.shape}
mod, params = relay.frontend.from_keras(model, shape_dict)

## we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)
print('------------original_mod----------')
f = open('./original_mod.txt','w')
print(func,file=f)
f.close()
######################################################################
# now compile the graph
target = "llvm"
with tvm.transform.PassContext(opt_level=0):
    graph, lib, params = relay.build(func, target, params=params)
# print('------------opt_level=1-----------')
# f = open('./opt_level0.json','w')
# print(graph,file=f)
# f.close()

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime

ctx = tvm.cpu(0)
dtype = "float32"
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input("data", tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)
print(tvm_output)
