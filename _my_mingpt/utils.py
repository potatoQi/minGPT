class CfgNode:
    """ 配置类 """

    # 允许通过一个字典或关键字参数来初始化对象属性
    def __init__(self, *args, **kwargs):    # args 是位置参数, kwargs 是关键字参数
        if len(args) == 1 and isinstance(args[0], dict):    # 用字典来更新属性
            self.__dict__.update(args[0])
        elif len(kwargs) != 0:    # 用关键字参数来更新字典
            self.__dict__.update(kwargs)
        elif len(args) == 0 and len(kwargs) == 0:   # 无参数
            pass
        else:
            print('Error: 传的参数要不就用一个字典, 要不就用关键字参数, 要不就啥都不传')

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

asd = {
    'a': 1,
    'b': 2,
}
C = CfgNode(q=3, w=4, e=CfgNode(asd), r=5)
print(C)