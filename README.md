本项目 fork 自 karpathy 大神的 [minGPT](https://github.com/karpathy/minGPT)，核心代码约 800 行。

是我 [练习 pytorch ](https://github.com/potatoQi/pytorch_learning) 项目的一个子项目。 

`minGPT` 为 karpathy 写的，`my_minGPT` 是我写的。

## Files tree
`minGPT` 库由以下四个主要文件组成：
1. mingpt/model.py：包含实际的 Transformer 模型定义。
2. mingpt/bpe.py：包含一个经过轻微重构的字节对编码器（Byte Pair Encoder），用于在文本和整数序列之间转换，方式与 OpenAI 的 GPT 相同。
3. mingpt/trainer.py：包含与 GPT 无关的 PyTorch 模板代码，用于训练模型。
4. mingpt/utils.py：工具文件，包括配置类、种子设置、日志记录

此外，`projects` 文件夹中还有若干示例和项目：
- projects/adder：从零开始训练一个 GPT 模型来进行加法运算，灵感来源于 GPT-3 论文中的加法部分。
- projects/chargpt：训练一个 GPT 模型，成为一个基于字符的语言模型，输入文本文件进行训练。
- generate.ipynb：展示如何加载一个预训练的 GPT2 模型，并根据给定的提示生成文本。
- demo.ipynb：展示了在笔记本格式下使用 GPT 模型和 Trainer 的最小示例，采用简单的排序任务。
- my_demo.ipynb 同上, 只不过换成我自己写的库

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
  - class generate
- 实现 `train.py`, 包括:
  - def get_default_config
  - def \_\_init\_\_
  - def run_train
  - def train_loop
  - def run_eval
  - def eval_loop
- 继续完成 demo.ipynb 的评估和推理