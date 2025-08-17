# README.md

## 项目简介
本项目实现了PD-HeteroGNN（Perplexity-Driven Heterogeneous Graph Neural Network），用于AIGC（人工智能生成内容）文本检测。  
核心思路是结合 困惑度（Perplexity）特征与TF–IDF 特征，构建文档–词异构图，并利用Heterogeneous Graph Transformer (HGT)进行分类，从而有效区分人写文本与机器生成文本。


## 环境依赖
请使用 Python 3.8+，推荐在虚拟环境或 Conda 中安装依赖。

```bash
conda create -n pdhetero python=3.8
conda activate pdhetero
```

安装主要依赖：
pip install torch==1.13.1
pip install torch-geometric==2.3.1
pip install scikit-learn==1.2.2
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install transformers==4.30.2
pip install tqdm
pip install jieba



## 数据准备

### 1. 数据集
本实验主要基于HC3-Bilingual数据集（中英文混合AIGC检测语料）。  
你需要准备一个CSV文件，包含以下字段：
`text`：文本内容
`label`：标签（0 = 人类文本，1 = AI生成文本）

例如：
text,label
"这是一个人工撰写的句子。",0
"This is a machine-generated sentence.",1

将数据文件命名为hc3_bilingual.csv并放置在项目根目录。


## 运行方法

### 训练模型
运行以下命令开始训练：
python PDHGN.py --mode train --csv hc3_bilingual.csv


### 评估模型
训练完成后，运行以下命令在测试集上评估：
python PDHGN.py --mode eval --csv hc3_bilingual.csv


### 主要参数说明
| 参数             | 说明                             | 默认值     |
| ------------ | ---------------------- | ------- |
| --mode       | 运行模式，可选 train/eval | 必填 |
| --csv            | 输入数据集路径                | 必填 |
| batch_size  | 批大小（需在代码中修改）| 32  |
| epochs       | 训练轮数（需在代码中修改） | 5 |



## 结果与复现
运行后会输出：
Accuracy, Precision, Recall, F1, ROC-AUC 等指标  
训练好的模型权重（如final_model.pth）


## 引用
如果你使用了本代码，请引用相关论文：  
Rui Peng. Perplexity-Driven HeteroGNN for AIGC Text Detection. 2025.
