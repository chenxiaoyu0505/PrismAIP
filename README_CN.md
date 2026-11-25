[English](README.md) · 简体中文

# PrismAIP

**抗炎肽（AIP）** 四视图预测器。每条肽由四个相互独立的模型分别打分，再通过学习得到的排名平均进行融合；整体不做端到端训练。

| 视图 | 关注的信息 | 模型 |
|---|---|---|
| `hand` | 694 个手工描述符 + 10 个两亲性指标 | ExtraTrees（1,500 棵树） |
| `gap4` | 容错 4-mer（每个窗口留空一位） | XGBoost |
| `k5` | 连续 5-mer | LightGBM |
| `tlora` | ESM-2 650M，在 q/k/v 上加数据对齐的低秩适配器 | ESM-2 + 适配器 + 位置头 |

每个视图的输出概率会映射到它在训练集折外（OOF）分布上的百分位排名；四个排名按在这些 OOF 分数上拟合的权重（Caruana 前向选择，27 轮）加权合并，再经 Platt 映射得到概率；判定阈值取 OOF 上 MCC 最优值。由于参考分布仅来自训练数据，融合分数不随本次输入文件中的其他行而改变。

---

## 安装

```bash
pip install -r requirements.txt
```

需要 Python 3.10+。只有 `tlora` 视图需要 GPU（推理约 1.5 GB 显存）；加 `--device cpu` 可在无 GPU 环境运行。

### ESM-2 主干（打分与重训都需要）

`tlora` 视图基于 `facebook/esm2_t33_650M_UR50D`。其权重**不在本仓库中**（约 2.6 GB），首次打分或重训时从 Hugging Face 拉取，因此首次运行需要联网。

如需提前下载（一次性，约 2.6 GB）：

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
m='facebook/esm2_t33_650M_UR50D'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
```

使用镜像或网络较慢时，先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # Hugging Face 镜像
export HF_HOME=/data/hf_cache               # 约 2.6 GB 的缓存位置
```

缓存完成后可**完全离线**运行，不会回连或下载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/predict.py --input my.csv
```

若主干无法加载，`predict.py` 会明确报错，而不是静默地只用三个视图打分；如仅需三个经典视图，可用 `--views hand,gap4,k5 --device cpu`（无需 ESM、无需 GPU、无需下载）。

## 对多肽打分

```bash
python scripts/predict.py --input example_input.csv
```

输入：含 `sequence` 列的 csv（其余列会原样保留）。输出：原有各行，加上每个视图的概率与融合结果：

| 列 | 含义 |
|---|---|
| `p_hand`, `p_gap4`, `p_k5`, `p_tlora` | 各视图概率 |
| `prismaip_score` | 融合排名分（权重作用于此） |
| `prismaip_prob` | 校准后概率 |
| `prismaip_call` | 推荐阈值下的 `AIP` / `non-AIP` |

常用参数：`--views hand,gap4,k5`（跳过神经视图）、`--device cpu`、`--out my_scores.csv`。

## 报告性能

留出测试集（`data/test1.csv`：817 条，262 条阳性）。数值为 3 个随机种子（0/1/2）
的均值 ± 标准差；每个种子在 `data/train1.csv` 上拟合分层 5 折交叉验证集成，
并在该测试集上评估一次。所有模型选择与校准仅使用训练数据。

| 视图 | AUROC | AUPR |
|---|---|---|
| hand | 0.783 ± 0.001 | 0.595 ± 0.001 |
| gap4 | 0.805 ± 0.003 | 0.621 ± 0.004 |
| k5 | 0.820 ± 0.003 | 0.662 ± 0.007 |
| tlora | 0.803 ± 0.021 | 0.601 ± 0.054 |
| **fused** | **0.8468 ± 0.0055** | **0.6796 ± 0.0111** |

`verify.py` 用随附的单套模型权重对 `test1` 重新打分，并核对记录的参考数值
（容差 2e-3）：

```bash
python scripts/verify.py
```

## 仓库内容

```
data/            train1.csv、test1.csv、manifest.json（sha256 与折划分）
weights/ml/      三个经典视图各一个模型，在完整 train1 上拟合（.xz）
weights/tlora/   五个折的增量 + A 的冻结 SVD 初始化
weights/fusion.json, fusion_ecdf.npz, oof.npz
scripts/         predict.py（入口）、verify.py、train_all.py 及各视图代码
```

## 重训

```bash
python scripts/train_all.py               # ML 权重、融合、期望分数
```

三个经典视图会在完整 `train1` 上重新拟合（即随附权重的配置）。融合权重始终基于 `weights/oof.npz` 中交叉验证的 OOF 分数拟合，绝不使用测试数据。

报告数字所用的交叉验证协议为 `StratifiedKFold(5, shuffle=True, random_state=42)`，按文件顺序作用于 `train1` 各行，并在种子 0/1/2 上重复；`scripts/ml_views.py` 中的 `train_folds()` 可复现各折集成。

重训需要已缓存 ESM-2 主干（见上文*安装*），在 GPU 上约需 3 分钟。

## 说明

* 序列按原样读取并转为大写；非标准残基在 k-mer 视图中被忽略。经典视图不做截断。

## 引用

若使用本代码，请引用 PrismAIP 论文（待补）与 ESM-2 主干（Lin et al., 2023）。适配器遵循 TLoRA（Lin et al., 2026）：A 由 `W0 @ C` 的前几个右奇异向量初始化并冻结，仅训练 B。

## 许可

MIT；见 `LICENSE`。
