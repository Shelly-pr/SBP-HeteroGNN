# README.md

## SBP-HeteroGNN
This repository contains the official implementation of the paper:**Streaming Bilingual Perplexity‑Driven HeteroGNN: A Heterogeneous Graph Transformer with Incremental Training for AIGC Text Detection**.
Accepted at the 9th International Conference on Computer Science and Artificial Intelligence (CSAI 2025), Beijing, China.

## Introduction
SBP-HeteroGNN is a lightweight heterogeneous graph neural network framework designed for **Chinese–English AIGC text detection**.  
Key features include:
- Hybrid tokenizer combining regex and Jieba for bilingual preprocessing.
- Perplexity-driven edge construction using GPT-2 signals.
- Heterogeneous Graph Transformer for robust feature fusion.
- Bilingual embeddings (GloVe + Tencent Word2Vec).
- Streaming incremental training for large-scale, real-time monitoring.

## Requirements
Python 3.8+

```bash
conda create -n pdhetero python=3.8
conda activate pdhetero
```

pip install torch==1.13.1
pip install torch-geometric==2.3.1
pip install scikit-learn==1.2.2
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install transformers==4.30.2
pip install tqdm
pip install jieba



## Dataset Preparation

### 1. HC3-Bilingual Dataset
The experiments are conducted on the **HC3-Bilingual** dataset (a Chinese–English mixed corpus for AIGC detection).  
You need to prepare a `.csv` file with the following columns:  
- `text`: raw text content  
- `label`: class label (`0 = human-written`, `1 = AI-generated`)  

**Example format:**  
```csv
text,label
"这是一个人工撰写的句子。",0
"This is a machine-generated sentence.",1
```

## Usage

### 1.Training
Run the following command to start training:
```bash
python SBPHGN.py --mode train --csv hc3_bilingual.csv
```

### 2.Evaluation
After training, evaluate the model on the test set using:
python SBPHGN.py --mode eval --csv hc3_bilingual.csv


### 3.Key Parameters
| Argument     | Description                                | Default  |
| ------------ | ------------------------------------------ | -------- |
| --mode       | Execution mode:train/eval                  | Required |
| --csv        | Path to input dataset                      | Required |
| batch_size   | Mini-batch size (modify directly in code)  | 32       |
| epochs       | Number of training epochs (modify in code) | 5        |



### 4.Results & Reproducibility
After running, the framework will output:
Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
Trained model weights, e.g. final_model.pth


If you use this code or find it helpful in your research, please cite the following paper:
**Rui Peng, Yuejin Zhang.** 
**Streaming Bilingual Perplexity-Driven HeteroGNN: A Heterogeneous Graph Transformer with Incremental Training for AIGC Text Detection.** 
In **Proceedings of the 9th International Conference on Computer Science and Artificial Intelligence (CSAI 2025)**, Beijing, China, 2025.

