import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)                   # random
    np.random.seed(seed)                # numpy
    torch.manual_seed(seed)             # torch-cpu
    torch.cuda.manual_seed_all(seed)    # torch-gpu


class CfgNode:
    """ 配置类 """

    # 通过关键字参数来初始化对象属性
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # print(obj) 时调用的函数实现
    def __str__(self):
        return self.str_helper(0)
    
    def str_helper(self, indent): # 递归打印
        res = []
        for (k, v) in self.__dict__.items():
            if not isinstance(v, CfgNode):
                res.append(f'{k}: {v}\n')
            else:
                res.append(f'{k}:\n')
                res.append(v.str_helper(indent + 1))
        res = [' ' * indent * 4 + entry for entry in res]
        return ''.join(res)

    # 通过字典更新配置
    def merge_from_dict(self, dict):
        self.__dict__.update(dict)

    # 将配置转换成字典返回
    def to_dict(self):
        return {
            k: v if not isinstance(v, CfgNode) else v.to_dict()
            for (k, v) in self.__dict__.items()
        }
