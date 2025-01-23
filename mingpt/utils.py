
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    """ 一个轻量级的配置类 (受到 yacs 库的启发) """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # obj.__dict__ 是一个字典, 管理着对象的属性名和属性值
        # 然后 obj.__dict__.update(kwargs) 就是将 kwargs 中的键值对更新到 obj.__dict__ 中
        # ** 在函数定义中, 表示将所有以关键字形式传递的参数捕获到一个字典中
        # ** 在函数调用中, 表示将字典中的键值对解包成关键字参数传递给函数
        # * 在函数定义中, 表示将所有以位置形式传递的参数捕获到一个元组中
        # * 在函数调用中, 表示将可迭代对象中的元素解包成位置参数传递给函数

    def __str__(self):
        # __str__(self) 是一个魔术方法, 即内置的方法, 在 print(obj) 或 str(obj) 会调用这个方法
        # 如果不重写它, 默认返回的是对象的内存地址, 例如 <__main__.CfgNode object at 0x7f7f7f7f7f7f>
        return self._str_helper(0)  # 从 0 级缩进开始

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        """ 需要一个助手来支持嵌套缩进，以实现美观的打印 """
        parts = []
        for k, v in self.__dict__.items():  # dict.items() 以列表形式返回字典中的所有键值对, 每一个键值对是一个元组
            if isinstance(v, CfgNode):  # 若 v 是 CfgNode 类型的实例 (instance: 实例)
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1)) # 递归打印
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts] # 给每行加上缩进
        return "".join(parts)   # str.join(list) 将 list 中的字符串连接成一个字符串, 间隔符是 str

    def to_dict(self):
        """ return a dict representation of the config """
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
            # 把 CfgNode 类型的属性转换成字典
            # dict.items() 以列表形式返回字典中的所有键值对, 每一个键值对是一个元组, k 是键, v 是值
            # 正常就是 k: v, 若 v 是 CfgNode 类型的实例, 就是 k: v.to_dict(), 递归调用
        }

    def merge_from_dict(self, d):
        # __init__ 那里是可以直接通过参数在初始化时去更新对象的属性 (a=1, b=CfgNode(c=2), d=3)
        # 但是这里可以通过字典的方式去更新对象的属性, 例如 obj.merge_from_dict({'a': 1, 'b': {'c': 2}, 'd': 3})
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
