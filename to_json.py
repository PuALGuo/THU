## to_json

import json

class Node(object):

    ## 别写在这里，会变成静态变量

    def __init__(self, name=None):
        
        self.info = {}
        self.info["name"] = name

    def __str__(self): ## change the print_out

        return self.info
        
    def __setitem__(self, key, value):

        self.info[key] = value

    def __getitem__(self, key):

        return self.info[key]
    
    def getinfo(self):

        return self.info

    def setinfo(self, node_info):

        self.info.update(node_info)

    # def __getattribute__(self,obj):

    #     try:
    #         return object.__getattribute__(self, obj)
    #     except:
    #         return None

class Graph(object):

    op_map = {
        
        ## general
        "image.resize":"upsample",
        "nn.conv2d":"conv2d",
        "nn.bias_add":"add",
        "nn.relu":"relu",

        "add":"add"

    }

    def __init__(self):

        self.info = []
    
        self.nodes_info = {} ## node 数据信息
        self.nodes_map = {}  ## num - Node 映射
        self.op_count = {} ## count for different op 
        
        print("[Graph] Create the data-graph")

    # def __str__(self): ## change the print_out

    #     return self.info

    def getinfo(self):

        return self.info

    def readmodel(self, path):

        f = open(path,"r")
        filelines = f.readlines()
        for fileline in filelines:
            self.func2node(fileline)
        f.close()
        return self.info

    def relu(self, op, num, inputs, options_1, options_2, outputs):

        assert(self.nodes_map[inputs[0]]["operation"] == "conv")
        self.nodes_map[num] = self.nodes_map[inputs[0]]
        self.nodes_map[num]["activation_type"] = "Relu"
        self.nodes_info[num] = self.nodes_info[inputs[0]]
        return None

    def add(self, op, num, inputs, options_1, options_2, outputs):

        if self.nodes_map[inputs[0]]["operation"] == "conv":
            assert(not(self.nodes_map[inputs[0]]["load_bias"]))
            self.nodes_map[num] = self.nodes_map[inputs[0]]
            self.nodes_map[num]["load_bias"] = True
            self.nodes_map[num]["bias_shift"] = 5 ## 默认参数
            self.nodes_map[num]["bias_dtype"] = "int8"
            self.nodes_map[num]["bias_log2scale"] = 9
            self.nodes_info[num] = self.nodes_info[inputs[0]]
            return None
        else:
            self.op_count.setdefault(op,[]).append(num)
            node = Node(op + str(len(self.op_count[op])-1))
            self.nodes_map[num] = node
            node_info = {
                "name":op + str(len(self.op_count[op])-1),
                "operation":op
                }
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
            node_info["pl_name"] = self.nodes_map[inputs[0]]["name"]
            node_info["add_name"] = self.nodes_map[inputs[1]]["name"]
            shape, dtype = self.nodes_info[inputs[0]]
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
                node_info["previous_layer"].append(self.nodes_map[inp]["name"])   
                if self.nodes_map[inp]["next_layer"][0] == "endpoint":
                    self.nodes_map[inp]["next_layer"] = []
                self.nodes_map[inp]["next_layer"].append(node_info["name"])
            node.setinfo(node_info)
            
            return node

    def upsample(self, op, num, inputs, options_1, options_2, outputs):

        self.op_count.setdefault(op,[]).append(num)
        node = Node(op + str(len(self.op_count[op])-1))
        self.nodes_map[num] = node
        node_info = {
            "name":op + str(len(self.op_count[op])-1),
            "operation":op
            }
        default_info = {
            "input_log2scale": 7,
            "output_log2scale": 7,
            "next_layer": [
              "endpoint"
            ]
        }
        node_info.update(default_info)
        shape, dtype = self.nodes_info[inputs[0]]
        _, c, h, w = shape
        node_info["input_channel_num"] = c
        node_info["input_dtype"] = "int8" ##
        node_info["input_size"] =  {
            "height": h,
            "width": w
        }
        self.nodes_info[num] = outputs[0]
        shape, dtype = self.nodes_info[num]
        _, c, h, w = shape
        node_info["output_channel_num"] = c
        node_info["output_dtype"] = "int8" ##
        node_info["output_size"] =  {
            "height": h,
            "width": w
        }
        assert(self.nodes_info[num][0][2] % self.nodes_info[inputs[0]][0][2] == 0)
        node_info["upsample_size"] = self.nodes_info[num][0][2] // self.nodes_info[inputs[0]][0][2]
        ## infer for previous_layer
        ''' 
        对上下层的推理可能有点问题，还需要更多的例子
        '''
        node_info["previous_layer"] = []
        for inp in inputs:
            node_info["previous_layer"].append(self.nodes_map[inp]["name"])   
            if self.nodes_map[inp]["next_layer"][0] == "endpoint":
                self.nodes_map[inp]["next_layer"] = []
            self.nodes_map[inp]["next_layer"].append(node_info["name"])
        node_info["mode"] = {
            '"nearest_neighbor"' : "nearest"
        }.get(options_2["method"], None) ## 欸 我这边None的位置写个func是可以被调用的，但assert(0)不行，但用func包起来就可以 
                                         ## 但我用匿名函数写就不运行
        if node_info["mode"] == None:
            print("[func2node] Undefined method ", options_2["method"])
            assert(0)
        node.setinfo(node_info)

        return node

    def conv(self, op, num, inputs, options_1, options_2, outputs):

        self.op_count.setdefault(op,[]).append(num)
        node = Node(op + str(len(self.op_count[op])-1))
        self.nodes_map[num] = node
        node_info = {
            "name":op + str(len(self.op_count[op])-1),
            "operation":op
            }
        
        default_info = {
            "activation_type": "None",
            "input_log2scale": 7,
            "weight_log2scale": 7,
            "output_log2scale": 7,
            "output_shift": 7,
            "load_bias": False,
            "next_layer": [
              "endpoint"
            ]
        }
        node_info.update(default_info)
        shape, dtype = self.nodes_info[inputs[0]]
        _, c, h, w = shape
        node_info["input_channel_num"] = c
        node_info["input_dtype"] = "int8" ##
        node_info["input_size"] =  {
            "height": h,
            "width": w
        }
        shape, dtype = self.nodes_info[inputs[1]] ## weight
        _, _, h, w = shape
        node_info["weight_dtype"] = "int8" ##
        node_info["kernel_size"] =  {
            "height": h,
            "width": w
        }
        self.nodes_info[num] = outputs[0]
        shape, dtype = self.nodes_info[num]
        _, c, h, w = shape
        node_info["output_channel_num"] = c
        node_info["output_dtype"] = "int8" ##
        node_info["output_size"] =  {
            "height": h,
            "width": w
        }
        ## infer for previous_layer
        ''' 
        对上下层的推理可能有点问题，还需要更多的例子
        '''
        node_info["previous_layer"] = []
        for inp in inputs:
            node_info["previous_layer"].append(self.nodes_map[inp]["name"])   
            if self.nodes_map[inp]["next_layer"][0] == "endpoint":
                self.nodes_map[inp]["next_layer"] = []
            self.nodes_map[inp]["next_layer"].append(node_info["name"])
        ##
        padding = options_1["padding"]
        padding = padding[padding.find("[")+1:padding.find("]")].split(", ")
        padding = [int(x) for x in padding]
        top, bottom, left, right = [int(x) for x in padding]
        node_info["padding"] = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right
        }
        stride = options_1.get("stride",'[1, 1]')
        stride = stride[stride.find("[")+1:stride.find("]")].split(", ")
        stride = [int(x) for x in stride]
        h, w = [int(x) for x in stride]
        node_info["stride"] = {
            "height": h,
            "width": w,
        }

        node.setinfo(node_info)

        return node

    def func2node(self, fileline):

        if fileline.startswith("fn"):
            start = 0
            end = len(fileline)
            index = fileline.find("%",start,end)
            while not(index==-1):
                name = fileline[index:fileline.find(":",index,end)].lstrip().rstrip()
                self.nodes_map[name] = {
                    "name":"input",
                    "next_layer": [
                    "endpoint"
                    ]
                    }
                shape = fileline[fileline.find("(",index,end)+1:fileline.find(")",index,end)].split(", ")
                shape = tuple([int(x) for x in shape])
                dtype = fileline[fileline.find(")")+2:fileline.find("]")].lstrip().rstrip()
                
                self.nodes_info[name] = [shape, dtype]
                start = fileline.find("]",index,end)
                index = fileline.find("%",start,end)
            return True
        if fileline.startswith("}"):
            return True

        fileline.lstrip().rstrip()
        start = 0
        end = len(fileline)
    
        ## index
        index = fileline.find("=",start,fileline.find("(",start,end))
        if index != -1:
            num = fileline[start:index].lstrip().rstrip()
            start = index + 1
        else:
            num = "%" + str(len(self.nodes_map.keys()))

        ## op
        index = fileline.find("(",start,end)
        if index == -1:
            print("[func2node] There is a blank line with no op")
            return False
        op = fileline[start:index].lstrip().rstrip()
        if self.op_map.get(op):
            op = self.op_map[op]
        else:
            print("[func2node] There is no translation for %s"%(op))
            return False
        start = index + 1

        ## input
        index = fileline.find(")", start, end)
        params = fileline[start:index]
        import re
        inputs = re.findall(r"%[^,()]+", params)
        options_1 = re.findall(r"([a-zA-Z_]+)=(\[[^\[]+\])", params)  ## 形如 size=[128, 128]
        options_1 = {x:y for (x,y) in options_1}
        options_2 = re.findall(r"([a-zA-Z]+)=([^,\[\])]+)", params)  ## 形如 method="nearest_neighbor" 
        options_2 = {x:y for (x,y) in options_2}
        start = index + 1
        
        ## output
        outputs = re.findall(r"ty=Tensor\[(.*), (.*)\]", fileline[start:])
        shape, dtype = outputs[0]
        shape = shape[shape.find("(")+1:shape.find(")")].split(", ")
        shape = [int(x) for x in shape]
        outputs[0] = (shape, dtype)

        ## create the node
        if op == "relu":
            node = self.relu(op, num, inputs, options_1, options_2, outputs)
        elif op == "add":
            node = self.add(op, num, inputs, options_1, options_2, outputs)
        elif op == "upsample":
            node = self.upsample(op, num, inputs, options_1, options_2, outputs)
        elif op == "conv2d":
            op = "conv"
            node = self.conv(op, num, inputs, options_1, options_2, outputs)

        if node:
            self.info.append(node.getinfo())
        
        return True

if __name__ == "__main__":
    
    import json

    path = './out/torch.txt'

    graph = Graph()

    graph.readmodel(path)

    with open('./out/ref.json','w') as f:
        tmp = json.load(open('./net_def.json','r'))
        print(json.dumps(tmp, sort_keys=True, indent=4, separators=(',', ': ')),file=f)
    with open('./out/mod.json','w') as f:
        print(json.dumps(graph.getinfo(), sort_keys=True, indent=4, separators=(',', ': ')),file=f)

