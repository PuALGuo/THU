## to_json

import json

class Node(object):

    info = {}

    def __init__(self, name=None):
        self.info["name"] = name

    def __str__(self): ## change the print_out
        return info
        
    def __setitem__(self, key, value):
        info[key] = value

    def __getitem__(self, key):
        return info[key]
    
    def getinfo(self):
        return info

    def setinfo(self, **kwargs):
        info.update(kwargs)

    # def __getattribute__(self,obj):

    #     try:
    #         return object.__getattribute__(self, obj)
    #     except:
    #         return None

class Graph(jobject):
    
    info = []
    
    nodes_info = {} ## node 数据信息
    nodes_map = {}  ## num - Node 映射
    op_count = {} ## count for different op 

    op_map = {
        
        ## general
        "image.resize":"upsample"
        "nn.conv2d":"conv2d"
        "nn.bias_add":"add"
        "nn.relu":"relu"

        "add":"add"

    }

    def __init__(self):
        print("[Graph] Create the data-graph")

    def __str__(self): ## change the print_out
        return self.info

    def readmodel(self, path):
        f = open(path,"r")
        filelines = f.readlines()
        for fileline in filelines:
            func2node(fileline)
        f.close()
        return self.info

    def relu(self, num, inputs, options_1, options_2, outputs):
        assert(self.nodes_map[inputs[0]]["operation"] == "conv")
        nodes_map[op] = nodes_map[inputs[0]]
        nodes_map[op]["activation_type"] = "Relu"

    def add(self, num, inputs, options_1, options_2, outputs):
        if nodes_map[inputs[0]]["operation"] == "conv":
            assert(not(nodes_map[op].get("load_bias", False)))
            nodes_map[op] = nodes_map[inputs[0]]
            nodes_map[op]["load_bias"] = true
            nodes_map[op]["bias_shift"] = 5 ## 默认参数
        else:
            op_count.setdefault(op,[]).append(num)
            node = Node(op + str(len(op_count[op])-1))
            node_info = {"operation":op}
            default_info = {
                "activation_type": "None",
                "pl_log2scale": 7,
                "pl_shiftbit": 2,
                "add_log2scale": 9,
                "add_shiftbit": 0,
                "output_log2scale": 6,
                "output_shift_bit": 3,
                "previous_layer": [
                  "input"
                ],
                "next_layer": [
                  "endpoint"
                ]
                }
            node_info.update(default_info)
            node_info["pl_name"] = nodes_map[inputs[0]]["name"]
            node_info["add_name"] = nodes_map[inputs[1]]["name"]
            shape, dtype = nodes_info[inputs[0]]
            _, c, h, w = shape
            node_info["input_channel_num"] = c
            node_info["input_size"] =  {
                "height": h,
                "width": w
            }
            node_info["dtype"] = "int8"  ## 应该用的是dtype，但我在tvm里暂时还没有加量化
            ## infer for previous_layer
            ''' 
            对上下层的推理可能有点问题，还需要更多的例子
            '''
            node_info["previous_layer"] = []
            for inp in inputs:
                node_info["previous_layer"].append(nodes_map[inp]["name"])   
                if nodes_map[inp]["next_layer"][0] = "endpoint":
                    nodes_map[inp]["next_layer"] = []
                nodes_map[inp]["next_layer"].append(node_info["name"])
            node.setinfo(node_info)
    
    def func2node(fileline):

        if fileline.startswith("fn"):
            start = 0
            end = len(fileline)
            index = fileline.find("%",start,end)
            while not(index==-1):
                name = fileline[index:fileline.find(":",start,end)].split()
                shape = fileline[fileline.find("(",start,end)+1:fileline.find(")",start,end)].split(", ")
                shape = tuple([int(x) for x in shape])
                dtype = fileline[fileline.find(")")+2:fileline.find("]")].split()
                
                nodes_info[name] = [shape, dtype]
                start = fileline.find("]")
            return True
        if fileline.startswith("}"):
            return True

        fileline.lstrip().rstrip()
        start = 0
        end = len(fileline)
    
        ## index
        index = fileline.find("=",start,end)
        if index != -1:
            num = fileline[start:index].split()
            start = index + 1
        else:
            num = "%" + str(len(node_map.keys()))

        ## op
        index = fileline.find("(",start,end)
        if index == -1:
            print("[func2node] There is a blank line with no op")
            return False
        op = fileline[start:index].split
        if op_map.get(op):
            op = op_map[op]
        else:
            print("[func2node] There is no translation for %s"%(op))
            return False
        start = index + 1

        ## input
        index = fileline.find(")", start, end)
        params = fileline[start:index]
        import re
        inputs = re.findall(r"(%[^,()]+), ", params)
        options_1 = re.findall(r"([a-zA-Z_]+)=(\[[^\[]+\])", params)  ## 形如 size=[128, 128]
        options_1 = {x:y for (x,y) in options_1}
        options_2 = re.findall(r"([a-zA-Z]+)=([^,\[\])]+)", params)  ## 形如 method="nearest_neighbor" 
        options_2 = {x:y for (x,y) in options_1}
        start = index + 1
        
        ## output
        outputs = re.findall(r"ty=Tensor\[(.*), (.*)\]", fileline[start:])
        
        ## create the node
        if op == "relu":
            self.relu(num, inputs, options_1, options_2, outputs)
        elif op == "add":

        elif op == "upsample":
            op_count.setdefault(op,[]).append(num)
            node = Node(op + str(len(op_count[op])-1))
            node_info = {"operation":op}
            default_info = {
                "input_log2scale": 7,
                "output_log2scale": 7,
            }
            node_info.update(default_info)
            shape, dtype = nodes_info[inputs[0]]
            _, c, h, w = shape
            node_info["input_channel_num"] = c
            node_info["input_dtype"] = "int8" ##
            node_info["input_size"] =  {
                "height": h,
                "width": w
            }
            shape, dtype = nodes_info[outputs[0]]
            _, c, h, w = shape
            node_info["output_channel_num"] = c
            node_info["output_dtype"] = "int8" ##
            node_info["output_size"] =  {
                "height": h,
                "width": w
            }
            assert(nodes_info[inputs[0]][0][2] % nodes_info[outputs[0]][0][2] == 0)
            node_info["upsample_size"] = nodes_info[inputs[0]][0][2] // nodes_info[outputs[0]][0][2]
            ## infer for previous_layer
            ''' 
            对上下层的推理可能有点问题，还需要更多的例子
            '''
            node_info["previous_layer"] = []
            for inp in inputs:
                node_info["previous_layer"].append(nodes_map[inp]["name"])   
                if nodes_map[inp]["next_layer"][0] = "endpoint":
                    nodes_map[inp]["next_layer"] = []
                nodes_map[inp]["next_layer"].append(node_info["name"])
            node_info["mode"] = {
                "nearest_neighbor" : "upsample"
            }.get(options_2["method"], None) ## 欸 我这边None的位置写个func是可以被调用的，但assert(0)不行，但用func包起来就可以 
                                             ## 但我用匿名函数写就不运行
            if node_info["mode"] == None:
                print("[func2node] Undefined method ", options_2["method"])
        elif op == "conv":


    
        # op_count.setdefault(op,[]).append(num)
        # node = Node(op + str(len(op_count[op])-1))
        # node_info = {"operation":op}
        # ## default params
        # node_info["upsample_size"] = 2
        # node_info["input_log2scale"] = 7
        # node_info["output_log2scale"] = 7

if __name__ == "__main__":
    