本项目 fork 自 karpathy 大神的 [minGPT](https://github.com/karpathy/minGPT)，核心代码约 800 行。

是我 [练习 pytorch ](https://github.com/potatoQi/pytorch_learning) 项目的一个子项目。 

`minGPT` 为 karpathy 写的，`_my_minGPT` 是我写的。到时候测试我写的对不对的时候，可以把我的文件名改为 `minGPT` 即可。

## Library Installation
```bash
conda activate xxx
cd minGPT
pip install -e .
```

## Usage
`demo.ipynb` 里有一个例子。

## Files tree
`minGPT` 库由以下三个主要文件组成：
1. mingpt/model.py：包含实际的 Transformer 模型定义。
2. mingpt/bpe.py：包含一个经过轻微重构的字节对编码器（Byte Pair Encoder），用于在文本和整数序列之间转换，方式与 OpenAI 的 GPT 相同。
3. mingpt/trainer.py：包含与 GPT 无关的 PyTorch 模板代码，用于训练模型。

此外，`projects` 文件夹中还有若干示例和项目：
- projects/adder：从零开始训练一个 GPT 模型来进行加法运算，灵感来源于 GPT-3 论文中的加法部分。
- projects/chargpt：训练一个 GPT 模型，成为一个基于字符的语言模型，输入文本文件进行训练。
- demo.ipynb：展示了在笔记本格式下使用 GPT 模型和 Trainer 的最小示例，采用简单的排序任务。
- generate.ipynb：展示如何加载一个预训练的 GPT2 模型，并根据给定的提示生成文本。