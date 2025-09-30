import os, sys
import math
import argparse

import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from torch_geometric.data import HeteroData, Batch
from torch.utils.data import DataLoader
from model import GNN, Classifier

import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import jieba

from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

# 记得train完保存一遍图，eval完记得再保存一遍图，因为共用文件夹

"""# ==== 强制使用宋体字体 ====
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"

# 1) 注册字体文件
fm.fontManager.addfont(font_path)

# 2) 设为首选 sans-serif
prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.family']     = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [prop.get_name()]
matplotlib.rcParams['axes.unicode_minus'] = False"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
from matplotlib import font_manager

# 1. 注册字体文件（路径根据你实际放置的位置调整）
font_manager.fontManager.addfont('/mnt/fonts/TIMES.TTF')
font_manager.fontManager.addfont('/mnt/fonts/TIMESI.TTF')
font_manager.fontManager.addfont('/mnt/fonts/TIMESBD.TTF')
font_manager.fontManager.addfont('/mnt/fonts/TIMESBI.TTF')

# 2. 更新 Matplotlib 全局配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 这里写你想用的字体名称
    'font.size': 8,
    'axes.unicode_minus': False
})

# —— 全局加载区 ——
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


plot_dirs = {
    "data_dist":       os.path.join(FIGURES_DIR, "data_distribution"),
    "perplexity":      os.path.join(FIGURES_DIR, "perplexity"),
    "perplexity_epoch":os.path.join(FIGURES_DIR, "perplexity_epoch"),
    "train_curve":     os.path.join(FIGURES_DIR, "train_curve"),
    "roc_pr":          os.path.join(FIGURES_DIR, "roc_pr"),
    "confusion":       os.path.join(FIGURES_DIR, "confusion"),
    "ablation":        os.path.join(FIGURES_DIR, "ablation"),
}
for d in plot_dirs.values():
    os.makedirs(d, exist_ok=True)


CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ——— 1) 加载英文 200d GloVe ———
def load_glove_txt(path):
    d, dim = {}, None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            w, vec = parts[0], [float(x) for x in parts[1:]]
            if dim is None: dim = len(vec)
            d[w] = torch.tensor(vec, dtype=torch.float)
    return d, dim

en_dict, embedding_dim = load_glove_txt('glove.6B.200d.txt')
# embedding_dim 此时应当是 200

# ——— 2) 加载腾讯中文二进制 200d ———
cn_wv = KeyedVectors.load_word2vec_format(
    'light_Tencent_AILab_ChineseEmbedding.bin', binary=True
)
cn_dict = {
    w: torch.tensor(cn_wv[w], dtype=torch.float)
    for w in cn_wv.key_to_index
}

# ——— 3) 合并：中文覆盖英文（若无冲突，可调顺序） ———
glove_dict = {**en_dict, **cn_dict}
print(f"[Info] Merged embeddings: vocab={len(glove_dict)}  dim={embedding_dim}")

# 中英文混合分类器
def mixed_tokenizer(text: str):
    # 按 英文单词 or 单个汉字块 拆分
    frags = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text)
    tokens = []
    for frag in frags:
        if re.match(r"[A-Za-z0-9_]+", frag):  # 英文或数字
            tokens.append(frag.lower())
        else:  # 汉字块
            tokens.extend(jieba.lcut(frag))  # 用 jieba 切分
    return tokens


# —————————————————————————————————————————
# 1. Perplexity 预处理
# —————————————————————————————————————————
class PerplexityCalculator:
    def __init__(self, model_dir: str, device: str = 'cpu'):
        """
        model_dir: 本地模型目录，比如英文 GPT2 在 gpt2_local，
                   中文 GPT2 在 ckpt/gpt2-chinese
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval()
        self.device = device

    def calc(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            loss = self.model(**enc, labels=enc['input_ids']).loss
        return float(torch.exp(loss))


# 1) 读全表 fit TF–IDF
df_full = pd.read_csv('/root/zhihu_ai_scraper_python/hc3_bilingual.csv')
# 随机打乱并按 70%/30% 划分
df_train, df_test = train_test_split(
    df_full, test_size=0.3, random_state=42, shuffle=True
)

# —— 插入：训练/测试样本分布柱状图 —— #
fig, ax = plt.subplots(figsize=(3, 2.5))
df_train['generated'].value_counts().sort_index().plot.bar(ax=ax, color='gray')

ax.set_title('Training Sample Distribution (0 = Human, 1 = AI)', fontsize=8, pad=6)
ax.set_xlabel('Label', fontsize=8)
ax.set_ylabel('Number of Samples', fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Human', 'AI'], fontsize=8)
ax.tick_params(labelsize=8)

fig.tight_layout()
fig.text(0.5, -0.07, 'Fig. 4. Training Sample Distribution (0 = Human, 1 = AI).',
         ha='center', fontsize=8)
fig.savefig(os.path.join(plot_dirs["data_dist"], "train_distribution.png"),
            dpi=300, bbox_inches='tight')
plt.close(fig)


tfidf = TfidfVectorizer(
    max_features=5000,
    tokenizer=mixed_tokenizer,  # ← 指定用混合分词器
    token_pattern=None  # ← 禁掉默认英文正则
)
tfidf.fit(df_train['text'].astype(str))

# 2) 初始化 perplexity 计算器
# pc = PerplexityCalculator(device=DEVICE)
en_pc = PerplexityCalculator(
    model_dir="/root/zhihu_ai_scraper_python/gpt2_local",
    device=DEVICE
)
zh_pc = PerplexityCalculator(
    model_dir="/root/zhihu_ai_scraper_python/gpt2_chinese_local",
    device=DEVICE
)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, perps, tfidf_vect, vocab=None, syntax_edges=None):
        assert len(texts) == len(labels) == len(perps)
        self.texts = texts
        self.labels = labels
        self.perps = perps
        self.tfidf = tfidf_vect
        self.vocab = vocab
        self.syntax = syntax_edges

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'perp': self.perps[idx]
        }


# —————————————————————————————————————————
# 2. 构建 HeteroData
# —————————————————————————————————————————
def build_hetero_graph(item,
                       tfidf_vect,
                       glove_dict,
                       embedding_dim,
                       vocab=None,
                       syntax_edges=None):
    text = item['text']
    label = item['label']

    # —— 根据文本中是否含有汉字来选择英文或中文 perplexity 计算器 ——
    def is_chinese(s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    perp = zh_pc.calc(text) if is_chinese(text) else en_pc.calc(text)

    data = HeteroData()

    # 1) 计算 lp_clamped
    lp = math.log(perp + 1.0)
    lp_clamped = max(0.0, min(lp / 10.0, 1.0))

    # 2) 构造 word.x（多语 fastText 或其他词向量字典）
    glove_vecs = []
    for word in tfidf_vect.get_feature_names_out():
        vec = glove_dict.get(word, torch.zeros(embedding_dim))
        glove_vecs.append(vec)
    data['word'].x = torch.stack(glove_vecs, dim=0)  # (V, D)

    # 3) 构建 text->word 边
    vec = tfidf_vect.transform([text]).tocoo()
    edge_index = torch.stack([
        torch.zeros_like(torch.tensor(vec.col)),
        torch.tensor(vec.col)
    ], dim=0)
    weights = torch.tensor(vec.data * lp_clamped, dtype=torch.float)
    data['text', 'contains', 'word'].edge_index = edge_index
    data['text', 'contains', 'word'].edge_weight = weights

    # 4) 计算词向量加权平均
    glove_matrix = data['word'].x  # (V, D)
    selected = glove_matrix[vec.col]  # (nnz, D)
    w = weights  # (nnz,)
    glove_emb = (selected * w.unsqueeze(1)).sum(dim=0) / (w.sum() + 1e-9)  # (D,)

    # 5) 拼接 [1.0, lp_clamped] 与 glove_emb
    orig = torch.tensor([1.0, lp_clamped], dtype=torch.float, device=glove_emb.device)  # (2,)
    text_feat = torch.cat([orig, glove_emb], dim=0)  # (D+2,)
    data['text'].x = text_feat.unsqueeze(0)  # (1, D+2)

    # 6) 可选：添加句法边
    if syntax_edges is not None:
        e = torch.tensor(syntax_edges, dtype=torch.long).t()
        data['word', 'syntax', 'word'].edge_index = e

    # 7) 标签
    data['text'].y = torch.tensor([label], dtype=torch.long)
    return data


# —————————————————————————————————————————
# 3. 模型封装
# —————————————————————————————————————————
class TextHGTClassifier(nn.Module):
    def __init__(self, metadata, in_dim, hid, n_rels, heads, layers, n_out):
        super().__init__()
        self.gnn = GNN(
            in_dim=in_dim,
            n_hid=hid,
            num_types=len(metadata[0]),
            num_relations=len(metadata[1]),
            n_heads=heads,
            n_layers=layers,
            dropout=0.2
        )
        self.cls = Classifier(hid, n_out)

    def forward(self, data: HeteroData):
        xt = data['text'].x
        xw = data['word'].x

        # 对齐文本和词节点特征维度后再拼接
        feat_dim_text = xt.size(1)
        feat_dim_word = xw.size(1)
        if feat_dim_word != feat_dim_text:
            pad_size = feat_dim_text - feat_dim_word
            if pad_size > 0:
                pad = torch.zeros(xw.size(0), pad_size, device=xw.device)
                xw = torch.cat([pad, xw], dim=1)
            else:
                pad = torch.zeros(xt.size(0), -pad_size, device=xt.device)
                xt = torch.cat([pad, xt], dim=1)

        x = torch.cat([xt, xw], dim=0)

        node_type = torch.cat([
            torch.zeros(xt.size(0), dtype=torch.long, device=xt.device),
            torch.ones(xw.size(0), dtype=torch.long, device=xw.device)
        ], dim=0)

        tw = data['text', 'contains', 'word'].edge_index
        ew = torch.cat([tw, tw.flip(0)], dim=1)
        et = torch.zeros(ew.size(1), dtype=torch.long, device=ew.device)
        etime = torch.zeros(ew.size(1), dtype=torch.long, device=ew.device)

        h = self.gnn(x, node_type, etime, ew, et)
        h_text = h[node_type == 0]
        return self.cls(h_text)


# —————————————————————————————————————————
# 4. 训练 与 推理
# —————————————————————————————————————————
def calc_batch_perplexity(texts, batch_size=8):
    perps = []
    for t in tqdm(texts, desc="Perplexity"):
        # 根据内容选择英文或中文计算器
        perp = zh_pc.calc(t) if re.search(r"[\u4e00-\u9fff]", t) else en_pc.calc(t)
        perps.append(perp)
    return perps


def train(csv_path, batch_sz=8, epochs=20):
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    labels = df['generated'].astype(int).tolist()
    # 新增：用于记录每轮的训练 loss 和验证 F1
    train_losses = []
    val_f1s = []

    # pc = PerplexityCalculator(device=DEVICE)
    perps = calc_batch_perplexity(texts, batch_size=batch_sz)

    # —— 在这里插入：全样本困惑度分布 —— #
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.boxplot(perps, notch=False, patch_artist=True, boxprops=dict(facecolor='lightgray'))

    ax.set_title('All Samples Perplexity Distribution', fontsize=8, pad=6)
    ax.set_xlabel('Sample Index', fontsize=8)
    ax.set_ylabel('Perplexity', fontsize=8)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.text(0.5, -0.07, 'Fig. 5. All Samples Perplexity Distribution.',
             ha='center', fontsize=8)
    fig.savefig(os.path.join(plot_dirs["perplexity"], "all_perplexity.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    X_tr, X_rem, y_tr, y_rem, p_tr, p_rem = train_test_split(
        texts, labels, perps, test_size=0.3, stratify=labels, random_state=42)

    # 第一轮 split 完成后，紧跟着插入：
    fig, ax = plt.subplots(figsize=(3, 2.5))
    import pandas as pd
    pd.Series(y_tr).value_counts().sort_index().plot.bar(ax=ax, color='gray')

    ax.set_title('Local Training Sample Distribution (0 = Human, 1 = AI)', fontsize=8, pad=6)
    ax.set_xlabel('Label', fontsize=8)
    ax.set_ylabel('Number of Samples', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'], fontsize=8)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.text(
        0.5, -0.07,
        'Fig. 6. Local Training Sample Distribution after First Split (0 = Human, 1 = AI).',
        ha='center', fontsize=8
    )
    fig.savefig(
        os.path.join(plot_dirs["data_dist"], "local_train_distribution.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)
    X_val, X_te, y_val, y_te, p_val, p_te = train_test_split(
        X_rem, y_rem, p_rem, test_size=0.5, stratify=y_rem, random_state=42)


    def collate_fn(batch):
        return Batch.from_data_list([
            build_hetero_graph(x, tfidf, glove_dict, embedding_dim)
            for x in batch
        ])

    train_loader = DataLoader(
        TextDataset(X_tr, y_tr, p_tr, tfidf),
        batch_size=batch_sz, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        TextDataset(X_val, y_val, p_val, tfidf),
        batch_size=batch_sz, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        TextDataset(X_te, y_te, p_te, tfidf),
        batch_size=batch_sz, shuffle=False, collate_fn=collate_fn)

    model = TextHGTClassifier((['text', 'word'], [('text', 'contains', 'word')]),
                              in_dim=embedding_dim + 2, hid=256, n_rels=1,
                              heads=2, layers=3, n_out=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2)

    best, patience = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch['text'].y.view(-1))
            opt.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg_loss = total / len(train_loader)
        train_losses.append(avg_loss)  # ← 记录 loss
        print(f"Epoch {ep} loss {avg_loss:.4f}")

        model.eval();
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                preds += model(batch).argmax(dim=1).cpu().tolist()
                trues += batch['text'].y.view(-1).cpu().tolist()
        f1 = f1_score(trues, preds, average='macro')
        val_f1s.append(f1)  # ← 记录 F1
        print(f"Val F1 {f1:.4f}")
        scheduler.step(f1)
        if f1 > best:
            best, patience = f1, 0
            torch.save(model.state_dict(), 'best.pth')
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping.")
                break

    # —— 绘制训练 Loss & 验证 F1 曲线 —— #
    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(val_f1s, label="Val F1", linestyle="--")
    ax2.set_ylabel("F1 Score")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dirs["train_curve"], "train_curve.png"))
    plt.close(fig)

    # 测试 & 保存
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            preds += model(batch).argmax(dim=1).cpu().tolist()
            trues += batch['text'].y.view(-1).cpu().tolist()
    print("Test F1", f1_score(trues, preds, average='macro'))
    torch.save(model.state_dict(), 'final_model.pth')
    print("Saved final_model.pth")


def train_streaming(csv_path,
                    batch_sz=8,
                    chunk_size=2000,  # 每块读取多少行
                    epochs_per_chunk=1):  # 每块数据训练多少 epoch
    # 1) 初始化模型、优化器、调度器
    model = TextHGTClassifier(
        (['text', 'word'], [('text', 'contains', 'word')]),
        in_dim=embedding_dim + 2,
        hid=256, n_rels=1,
        heads=2, layers=3, n_out=2
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2)
    chunk_losses = []

    # —— 断点恢复 ——
    ckpt_resume = os.path.join(CKPT_DIR, 'interrupt.pth')
    if os.path.exists(ckpt_resume):
        model.load_state_dict(torch.load(ckpt_resume, map_location=DEVICE))
        print(f"⚡ Resumed training from checkpoint: {ckpt_resume}")

    # 2) 定义和原来一样的 collate_fn
    def collate_fn(batch):
        return Batch.from_data_list([
            build_hetero_graph(x, tfidf, glove_dict, embedding_dim)
            for x in batch
        ])

    # 3) 按块读取并训练
    reader = pd.read_csv(csv_path, iterator=True, chunksize=chunk_size)
    val_f1_best = 0.0
    try:
        for chunk_idx, df_chunk in enumerate(reader):
            texts = df_chunk['text'].astype(str).tolist()
            labels = df_chunk['generated'].astype(int).tolist()

            # 6.1) 这块数据先算 perplexities
            perps = []
            for i in range(0, len(texts), batch_sz):
                batch_texts = texts[i:i + batch_sz]
                perps.extend(calc_batch_perplexity(batch_texts, batch_size=len(batch_texts)))

            # —— 插入每块困惑度分布可视化 —— #
            fig, ax = plt.subplots()
            ax.boxplot(perps)
            ax.set_title(f"Chunk {chunk_idx} Perplexity Distribution")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Perplexity")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dirs["perplexity_epoch"], f"chunk_{chunk_idx:03d}.png"))
            plt.close(fig)

            # 6.2) 构造 DataLoader，马上对这一块做 online 训练
            dataset = TextDataset(texts, labels, perps, tfidf)
            loader = DataLoader(dataset,
                                batch_size=batch_sz,
                                shuffle=True,
                                collate_fn=collate_fn)

            total_loss = 0.0
            model.train()
            for _ in range(epochs_per_chunk):
                for batch in loader:
                    batch = batch.to(DEVICE)
                    logits = model(batch)
                    loss = F.cross_entropy(logits, batch['text'].y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()

            print(f"Chunk {chunk_idx} trained, size={len(texts)} samples.")
            avg_loss = total_loss / (len(loader) * epochs_per_chunk)
            chunk_losses.append(avg_loss)
            print(f"Chunk {chunk_idx} avg loss: {avg_loss:.4f}")

            # —— 新增：保存当前模型状态 ——
            ckpt_path = os.path.join(CKPT_DIR, f'chunk_{chunk_idx}.pth')
            torch.save(model.state_dict(), ckpt_path)

            fig, ax = plt.subplots()
            ax.plot(chunk_losses, marker='o')
            ax.set_xlabel("Chunk Index")
            ax.set_ylabel("Average Loss")
            ax.set_title("Streaming Training: Chunk‐level Loss Curve")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dirs["train_curve"], "streaming_train_curve.png"))
            plt.close(fig)

    except KeyboardInterrupt:
        # 中断时立刻保存
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'interrupt.pth'))
        print("训练被中断，模型已保存到 checkpoints/interrupt.pth")
        raise

    # 最后保存模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("Streaming training done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer', 'eval'], default='train')
    parser.add_argument('--csv', default='/root/zhihu_ai_scraper_python/hc3_bilingual.csv')
    parser.add_argument('--texts', nargs='*', help='要预测的文本列表')
    args = parser.parse_args()

    # —— 1) 流式训练 ——
    if args.mode == 'train':
        train_streaming(
            csv_path=args.csv,
            batch_sz=8,
            chunk_size=2000,
            epochs_per_chunk=1
        )

    # —— 2) 批量推理 ——
    elif args.mode == 'infer':
        if not args.texts:
            print("请通过 --texts 指定要预测的文本列表")
        else:
            model = TextHGTClassifier(
                (['text', 'word'], [('text', 'contains', 'word')]),
                in_dim=embedding_dim + 2,
                hid=256, n_rels=1, heads=2, layers=3, n_out=2
            ).to(DEVICE)
            model.load_state_dict(torch.load('final_model.pth', map_location=DEVICE))
            model.eval()

            graphs = [
                build_hetero_graph(
                    {'text': t, 'label': 0, 'perp': 1.0},
                    tfidf, glove_dict, embedding_dim
                )
                for t in args.texts
            ]
            batch = Batch.from_data_list(graphs).to(DEVICE)
            with torch.no_grad():
                preds = model(batch).argmax(dim=1).cpu().tolist()
            for txt, p in zip(args.texts, preds):
                print(f"文本：{txt}\n预测标签：{p}")

    # —— 3) 测试集评估 ——
    else:  # args.mode == 'eval'
        # —— 1) 加载模型 ——
        model = TextHGTClassifier(
            (['text', 'word'], [('text', 'contains', 'word')]),
            in_dim=embedding_dim + 2,
            hid=256, n_rels=1, heads=2, layers=3, n_out=2
        ).to(DEVICE)
        model.load_state_dict(torch.load('final_model.pth', map_location=DEVICE))
        model.eval()

        # —— 2) 测试集
        # df_eval = df_full.sample(n=10000, random_state=42).reset_index(drop=True)
        df_eval = df_test
        X_te = df_eval['text'].astype(str).tolist()
        y_te = df_eval['generated'].astype(int).tolist()

        # —— 3) 重新计算这些样本的 Perplexity ——
        perps_te = calc_batch_perplexity(X_te, batch_size=8)


        # —— 4) 构造 DataLoader ——
        def collate_fn_eval(batch):
            return Batch.from_data_list([
                build_hetero_graph(x, tfidf, glove_dict, embedding_dim)
                for x in batch
            ])


        test_dataset = TextDataset(X_te, y_te, perps_te, tfidf)
        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False,
            collate_fn=collate_fn_eval, num_workers=0, pin_memory=True
        )

        # —— 5) 批量推理 & 打印指标 ——
        from sklearn.metrics import f1_score, precision_score, recall_score

        preds, trues, scores = [], [], []
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)  # 可能是 [B,2] 或 [2]
            # —— 这里收集正类（label=1）的预测概率 —— #
            # 处理 logits 形状为 [2] 或 [B, 2] 的两种情况
            if logits.dim() == 1:
                # 单样本，先扩展到 [1, 2]
                probs = torch.softmax(logits.unsqueeze(0), dim=1)[0, 1].unsqueeze(0)
            else:
                # 多样本，正常取第二列的概率
                probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().tolist())

            # 收集预测标签
            idx = logits.argmax(dim=-1)
            if idx.dim() == 0:
                preds.append(idx.item())
            else:
                preds += idx.cpu().tolist()

            # 收集真实标签
            trues += batch['text'].y.view(-1).cpu().tolist()

        print("Eval on test samples:")
        print("Test F1:    ", f1_score(trues, preds, average='macro'))
        print("Precision:  ", precision_score(trues, preds, average='macro'))
        print("Recall:     ", recall_score(trues, preds, average='macro'))

        # —— 1) Confusion Matrix ——
        cm = confusion_matrix(trues, preds)
        fig, ax = plt.subplots(figsize=(3, 2.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greys',
            cbar=False,
            ax=ax,
            annot_kws={'size': 8}
        )
        ax.set_title('Confusion Matrix', fontsize=8, pad=6)
        ax.set_xlabel('Predicted Label', fontsize=8)
        ax.set_ylabel('True Label', fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.text(0.5, -0.07, 'Fig. 1. Confusion Matrix.', ha='center', fontsize=8)
        fig.savefig(
            os.path.join(plot_dirs["confusion"], "confusion_matrix.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

        # —— 2) ROC Curve ——
        fpr, tpr, _ = roc_curve(trues, scores)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(fpr, tpr, linewidth=1)
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        ax.set_title('ROC Curve', fontsize=8, pad=6)
        ax.set_xlabel('False Positive Rate', fontsize=8)
        ax.set_ylabel('True Positive Rate', fontsize=8)
        ax.legend([f'SBP‑HeteroGNN (AUC={roc_auc:.3f})'], fontsize=6, loc='lower right')
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.text(0.5, -0.07, 'Fig. 2. ROC Curve.', ha='center', fontsize=8)
        fig.savefig(
            os.path.join(plot_dirs["roc_pr"], "roc_curve.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

        # —— 3) Precision–Recall Curve ——
        precision, recall, _ = precision_recall_curve(trues, scores)
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(recall, precision, linewidth=1)
        ax.set_title('Precision–Recall Curve', fontsize=8, pad=6)
        ax.set_xlabel('Recall', fontsize=8)
        ax.set_ylabel('Precision', fontsize=8)
        ax.legend(['SBP‑HeteroGNN'], fontsize=6, loc='lower left')
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.text(0.5, -0.07, 'Fig. 3. Precision–Recall Curve.', ha='center', fontsize=8)
        fig.savefig(
            os.path.join(plot_dirs["roc_pr"], "pr_curve.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

"""
Test F1:     0.9661509749899937
Precision:   0.9657701866476855
Recall:      0.9665358441123876
Computed ROC AUC = 0.9940
"""