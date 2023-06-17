<div align="center">
  <a href="http://tianshou.readthedocs.io"><img width="300px" height="auto" src="https://github.com/thu-ml/tianshou/raw/master/docs/_static/images/tianshou-logo.png"></a>
</div>

&nbsp;

[![PyPI](https://img.shields.io/pypi/v/tianshou)](https://pypi.org/project/tianshou/) [![Read the Docs](https://img.shields.io/readthedocs/tianshou)](https://tianshou.readthedocs.io/en/master) [![codecov](https://img.shields.io/codecov/c/gh/thu-ml/tianshou)](https://codecov.io/gh/thu-ml/tianshou) [![GitHub issues](https://img.shields.io/github/issues/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/issues) [![GitHub stars](https://img.shields.io/github/stars/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/stargazers) [![GitHub forks](https://img.shields.io/github/forks/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/network) [![GitHub license](https://img.shields.io/github/license/thu-ml/tianshou)](https://github.com/thu-ml/tianshou/blob/master/LICENSE)

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/baichuan-inc/baichuan-7B/blob/main/README_CN.md">中文</a>
    <p>
</h4>

**Zhijian** ([**执简**驭繁](https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81)) is a PyTorch-based lightweight framework for reusing pre-trained models and transferring them to new datasets. It offers a unified and flexible solution for popular methods such as parameter-efficient based, knowledge distillation based, nearest prototype based, regularization based, and model soup based reuse. Zhijian achieves state-of-the-art model reuse capabilities with a workflow that involves assembling addins, allocating training strategies, and aggregating model parameters. The provided interface methods include:

+ Assembling Addins with Parameter-Efficient Transfer Learning
  + [Adapter](https://arxiv.org/abs/1902.00751)
  + [LoRA](https://arxiv.org/abs/2106.09685)
  + [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
  + [Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823)
  + [Factor-Tuning](https://arxiv.org/abs/2212.03145)
  + [Convolutional Bypasses](https://arxiv.org/abs/2207.07039)
+ Allocating Training Strategies via Knowledge Distillation and Regularization
  + [FitNet](https://arxiv.org/abs/1412.6550)
  + [Relational-KD](https://arxiv.org/abs/1904.05068)
  + [L<sup>2</sup>SP](https://arxiv.org/abs/1802.01483)
  + [DELTA](https://arxiv.org/abs/1901.09229)
  + [Batch Spectral Shrinkage](https://proceedings.neurips.cc/paper/2019/hash/c6bff625bdb0393992c9d4db0c6bbe45-Abstract.html)
+ Aggregating Parameters from Model Soup and Merging
  + [Model Soup](https://arxiv.org/abs/2203.05482)
  + [WiSE-FT](https://arxiv.org/abs/2109.01903)

**Zhijian** also has the following highlights:

+ Extremely easy to **get started** and **customize**
  + [Get started](TODO) with a 10 minute blitz 
  + [Customize](TODO) any part with step-by-step instructions
  + [APIs](TODO) come with friendly guidelines
+ **Concise** things do big
  + **State-of-the-art** [VTAB benchmark](TODO) for TODO methods
  + only ~TODO lines of the code
  + Support incorporating method like building *LEGO* blocks [[here]](TODO)
  + Support any dataset and pre-trained model [[here]](TODO)
  + Support multi-GPU training [[here]](TODO)
  + Support both [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/) log tools [[here]](TODO)


> "Zhijian" in Chinese means handling complexity with concise and efficient methods. Given the variations in pre-trained models and the deployment overhead of full parameter fine-tuning, Zhijian represents a solution that is easily reusable, maintains high accuracy, and maximizes the potential of pre-trained models.
> 
> “执简驭繁”的意思是用简洁高效的方法驾驭纷繁复杂的事物。现有预训练模型体系结构差异大，全参数微调部署时间长，所以取名“执简”的意思是考虑一个易上手、快复用、稳精度的解决方案，最大限度激发预训练模型的能力。


## Quick Start

1. An environment with Python 3.7+ from
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"),
[venv](https://docs.python.org/3/library/venv.html), or [virtualenv](https://virtualenv.pypa.io/en/latest/).

2. Install Zhijian using pip:
   ```bash
   $ pip install zhijian
   ```
   For more details please click [installation instructions](TODO/INSTALL.md).

   + [Option] Install with the newest version through GitHub:
      ```bash
      $ pip install git+https://github.com/ZhangYikaii/zhijian.git@main --upgrade
      ```

3. Open your python console and type
   ```python
   import zhijian
   print(zhijian.__version__)
   ```
   If no error occurs, you have successfully installed Zhijian.


## Documentation

The tutorials and API documentation are hosted on [tianshou.readthedocs.io](https://tianshou.readthedocs.io/).

The example scripts are under [test/](https://github.com/thu-ml/tianshou/blob/master/test) folder and [examples/](https://github.com/thu-ml/tianshou/blob/master/examples) folder.

中文文档位于 [https://tianshou.readthedocs.io/zh/master/](https://tianshou.readthedocs.io/zh/master/)。

<style>
table {
  border-collapse: collapse;
}

td, th {
  border: 1px solid black;
}

td:nth-child(3), th:nth-child(3),
td:nth-child(4), th:nth-child(4) {
  border-left-width: 2px;
}
</style>

| 列1 | 列2 | 列3 | 列4 | 列5 |
| --- | --- | --- | --- | --- |
| 内容 | 内容 | 内容 | 内容 | 内容 |
| 内容 | 内容 | 内容 | 内容 | 内容 |
| 内容 | 内容 | 内容 | 内容 | 内容 |
| 内容 | 内容 | 内容 | 内容 | 内容 |

## Why Zhijian?

<h3 align="center">
    <p>
        Concise things do big
    <p>
</h3>


| Model Reuse Framework | GitHub Stars | # of Alg. <sup>(1)</sup> | # of Backbone <sup>(1)</sup> | Batch Training | RNN Support | Nested Observation | Backend    |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |--------------------------------| --------------------------------- | ------------------ | ------------------ | ---------- |
| [Baselines](https://github.com/openai/baselines)                   | [![GitHub stars](https://img.shields.io/github/stars/openai/baselines)](https://github.com/openai/baselines/stargazers)                         | 9                        | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
| [Stable-Baselines](https://github.com/hill-a/stable-baselines)     | [![GitHub stars](https://img.shields.io/github/stars/hill-a/stable-baselines)](https://github.com/hill-a/stable-baselines/stargazers)           | 11                       | :heavy_check_mark: (gym)       | :heavy_minus_sign: <sup>(2)</sup> | :heavy_check_mark: | :x:                | TF1        |
