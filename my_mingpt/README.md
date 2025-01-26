这是参考 `mingpt` 后我自己写的版本。

如果一个文件夹 `xxx` 下面有 `__init__.py` 这个文件，就是说明把 `xxx` 文件夹标记为 Python 包。到时候 `setup.py` 里的 `packages` 参数里就可以把 `xxx` 这个文件夹的名字写上去。这样到时候 `pip install . -e` 后就可以直接 `import xxx` 了。

## 任务
- 新建一个文件夹作为项目目录，并将其变为一个 python 包
- 实现 `utils.py/set_seed` 函数
  - 包括 random, np, torch-cpu, torch-gpu
- 实现 `utils.py/CfgNode` 配置类，要求具有下列功能:
   - 通过关键字参数初始化对象
   - 支持美观 print(obj) 输出
   - 支持通过字典更新配置
   - 支持将配置导出为字典
- 新建 demo.ipynb，完成随机数种子固定 + 数据集获取 + 模型创建 + 训练器创建
- 实现 `model.py`, 包括:
  - class minGPT
    - def get_default_config
    - def \_\_init\_\_
    - def \_init\_weights
    - def forward
  - class Block
  - class CausalSelfAttention
  - class NEWGELU
  - class forward
- 实现 `train.py`, 包括:
  - def get_default_config
  - def \_\_init\_\_
  - def run
  - def train_loop