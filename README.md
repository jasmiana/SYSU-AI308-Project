# BSARec 论文复现项目

本项目基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架，复现了论文 **"An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention"** 中提出的 BSARec 模型。

## 项目概述

BSARec (Beyond Self-Attention Recommendation) 是一种序列推荐模型，通过引入频域归纳偏置（Frequency-based Inductive Bias）来改进传统的自注意力机制。模型的核心思想是：

- **双分支架构**：结合 Self-Attention 和频域滤波（FFT-based Frequency Rescaling）
- **动态混合**：使用可学习的权重 `alpha` 平衡两个分支的贡献
- **频域处理**：通过傅里叶变换分离高频（短期兴趣）和低频（长期兴趣）信号

## 项目结构

```markdown
TASK/
├── ReChorus/                    # ReChorus 框架核心代码
│   ├── src/
│   │   ├── main.py             # 主入口文件
│   │   ├── models/
│   │   │   └── BSARec.py       # BSARec 模型实现（核心代码）
│   │   ├── helpers/             # Reader 和 Runner 模块
│   │   └── utils/               # 工具函数
│   ├── data/                    # 数据集目录
│   │   └── ML-1M/               # MovieLens-1M 数据集
│   ├── process_ml1m_final.py    # ML-1M 数据处理脚本（最终版本）
│   └── process_split_ml1m.py    # ML-1M 数据切分脚本（带负样本生成）
│   ├── requirements.txt        # Python 依赖
│   └── README0.md              # ReChorus 原始文档
├── log/                         # 训练日志（自动生成）
└── model/                       # 保存的模型文件（自动生成）
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+ (建议使用 CUDA 版本以加速训练)
- 其他依赖见 `ReChorus/requirements.txt`

### 安装步骤

1. **克隆或下载项目**

```bash
cd TASK/ReChorus
```

2. **安装依赖**

```bash
# PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install -r requirements.txt
```

3. **准备数据集**

本项目使用 Grocery_and_Gourmet_Food 数据集（原项目集成）和 MovieLens-1M 数据集。数据预处理有两种方式：

**方式一：使用 `process_ml1m_final.py`（简单版本）**
```bash
python process_ml1m_final.py
```

**方式二：使用 `process_split_ml1m.py`（带负样本生成，用于留一法评估）**
```bash
python process_split_ml1m.py
```

处理后的数据将保存在 `data/ML-1M/` 目录下，包含：
- `train.csv`: 训练集
- `dev.csv`: 验证集
- `test.csv`: 测试集

**注意**：原始数据文件 `ratings.dat` 需要放在 `data/ML-1M/raw/ml-1m/` 目录下。可以从 [MovieLens 官网](https://grouplens.org/datasets/movielens/1m/) 下载。

### 运行训练

**基本命令格式：**

```bash
cd ReChorus
python src/main.py --model_name BSARec --dataset ML-1M [其他参数]
```

**完整示例（Windows 环境）：**

```bash
python src/main.py \
    --model_name BSARec \
    --dataset ML-1M \
    --emb_size 64 \
    --num_layers 2 \
    --num_heads 2 \
    --alpha 0.7 \
    --c 5 \
    --beta_init 0.0 \
    --lr 0.001 \
    --l2 0 \
    --epoch 50 \
    --batch_size 256 \
    --history_len 20 \
    --num_workers 0 \
    --gpu 0
```

**关键参数说明：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--alpha` | 归纳偏置权重（0-1），越大越依赖频域滤波 | 0.3-0.9 |
| `--c` | 低频截止点，控制频域分离的阈值 | 1-9 |
| `--beta_init` | 高频缩放因子的初始值 | 0.0 |
| `--num_heads` | 自注意力头数 | 2-4 |
| `--num_layers` | 编码器层数 | 2 |
| `--emb_size` | 嵌入维度 | 64 |
| `--history_len` | 序列最大长度 | 20-50 |
| `--num_workers` | 数据加载进程数（Windows 建议设为 0） | 0 |

**不同数据集的推荐参数：**

- **ML-1M（稠密/长序列）**：`--alpha 0.7 --c 5`
- **稀疏数据集**：`--alpha 0.3 --c 5`

### 运行基线模型对比

为了验证 BSARec 的效果，可以运行以下基线模型：

```bash
# SASRec（Transformer 基线）
python src/main.py --model_name SASRec --dataset ML-1M --emb_size 64 --lr 5e-4 --epoch 50 --num_workers 0 --alpha 0.3 --c 9 --history_len 50 --num_heads 4

# GRU4Rec（RNN 基线）
python src/main.py --model_name GRU4Rec --dataset ML-1M --emb_size 64 --lr 5e-4 --epoch 50 --num_workers 0 --alpha 0.3 --c 9 --history_len 50 --num_heads 4
```

## 实验结果

实验结果保存在 `log/` 目录下，每个实验会生成：
- `.txt` 文件：训练过程的详细日志
- `.csv` 文件：最终推荐结果（Top-K 列表）

**评估指标：**
- **HR@K** (Hit Rate): 命中率，推荐列表中是否包含用户真正交互的物品
- **NDCG@K** (Normalized Discounted Cumulative Gain): 归一化折损累计增益，考虑命中物品的位置

## 核心代码说明

### BSARec 模型架构

模型位于 `ReChorus/src/models/BSARec.py`，主要包含三个类：

1. **BSARec**: 主模型类，继承自 `SequentialModel`
   - 实现序列嵌入、位置编码
   - 堆叠多个 BSALayer
   - 动态提取最后一个有效 item 的 hidden state

2. **BSALayer**: 编码器层
   - **分支1**：Self-Attention（标准 Transformer）
   - **分支2**：Inductive Bias（频域滤波）
   - **混合**：`alpha * IB + (1-alpha) * SA`
   - **FFN**：前馈网络

3. **FrequencyRescaler**: 频域重缩放模块
   - 使用 FFT 将序列转换到频域
   - 分离低频（`c` 以下）和高频（`c` 以上）分量
   - 对高频分量应用可学习的缩放因子 `beta`
   - 通过 IFFT 转换回时域

### 关键修复

代码中包含了重要的 bug 修复：
- **动态序列提取**：使用 `gather` 操作正确提取每个用户最后一个有效 item 的表示，而不是简单地取序列末尾（可能包含 padding）

## 常见问题

### Windows 环境问题

1. **多进程错误**：将 `--num_workers` 设为 `0`
2. **路径问题**：确保使用正确的路径分隔符（代码已处理）
3. **DLL 冲突**：如遇到 CUDA DLL 问题，检查 PyTorch 和 CUDA 版本兼容性

### 数据问题

1. **找不到数据文件**：确保 `ratings.dat` 在 `data/ML-1M/raw/ml-1m/` 目录下
2. **数据格式错误**：检查 CSV 文件是否为 tab 分隔（`\t`）

### 训练问题

1. **内存不足**：减小 `--batch_size` 或 `--history_len`
2. **收敛慢**：调整学习率 `--lr` 或增加训练轮数 `--epoch`
3. **结果异常**：检查超参数设置，特别是 `--alpha` 和 `--c`

## 参考文献

- **BSARec 论文**：An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention
- **ReChorus 框架**：[THUwangcy/ReChorus](https://github.com/THUwangcy/ReChorus)

## 作者

本项目为论文复现作业，基于 ReChorus 2.0 框架实现。

---

**提示**：首次运行建议先在小数据集或少量 epoch 上测试，确认环境配置正确后再进行完整训练。