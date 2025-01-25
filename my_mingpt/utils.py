class CfgNode:
    """ 配置类 """

    # 通过关键字参数来初始化对象属性
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            self.__dict__.update(args[0])
        elif len(kwargs) != 0:
            self.__dict__.update(kwargs)
        elif len(args) == 0 and len(kwargs) == 0:
            pass
        else:
            assert 0, '参数输入格式错误'

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

    # 将配置转换成字典返回
    def to_dict(self):
        return {
            k: v if not isinstance(v, CfgNode) else v.to_dict()
            for (k, v) in self.__dict__.items()
        }
