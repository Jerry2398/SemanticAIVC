# PerturbEnergy — 用能量模型增强 latent space 中的 perturbation-specific 信息

论文 (`PerturbEnergy.pdf`) 的实现:在扰动隐空间上引入**条件能量先验**
`p_α(z_a|z_b,a) = p_0(z_a)exp(−E_α(z_a,z_b,a))/F_α(z_b,a)`,通过 Energy-ELBO
(Eq. 8) + Langevin 负采样 (Eq. 15) + 对比式梯度 (Eq. 12) 直接增强 `a` 与 `z_a`
的依赖(Theorem 2:提升 `I(A;Z_a|Z_b,C)` 的变分下界)。

## 模型(全部 scGen 式 MLP,hidden=512、depth=2,与既有 Tahoe VAE 统一)

| 模块 | 映射 | 说明 |
|---|---|---|
| `BasalEncoder` `q_φβ(z_b\|x,c)` | x(5000 HVG)+ c(cell line 嵌入) → (μ_b, logσ²_b) | 基础细胞状态 |
| `PertEncoder` `q_φα(z_a\|z_b,a)` | z_b + a(drug 嵌入) → (μ_a, logσ²_a) | 扰动表示 |
| `Generator` `p_θ(x\|z_b,z_a)` | [z_b,z_a] → x̂(线性输出,log-norm 表达) | 与 scGen/我们的 VAE 同口径 |
| `EnergyNet` `E_α(z_a,z_b,a)` | [z_a,z_b,a 嵌入] → 标量 | **能量越低 = 该扰动表示与 (z_b,a) 越兼容** |

**Embedding 只有两张表**(`cond_emb`、`pert_emb`),由顶层模型持有、被两个 encoder 与
energy 网络**共享**;它们属于 VAE 参数组,能量更新时以 detach 形式传入,两个相位互不干扰。
两个隐变量的先验都是固定的 `N(0,I)`(无学习型 prior 模块)。

**forward**:`z_b` → 采 **M 个候选 `z_a`** → **能量为每个候选打分** → 按 `min`(能量最低)
或 `softmax`(p ∝ exp(−E/τ),归一化)**选出一个** → 解码。即能量直接参与前向选择。

**训练(Algorithm 1,每步)**:
① 前向(能量只读,用于选择 z_a);
② **相位 A — 更新 α**:正样本 = 被选中的 `z_a`;负样本 = **从 batch 内另一个细胞的 `z_a` 出发**
   跑 Langevin(比白噪声起点信息量高得多);损失用 **hinge**
   `max(0, margin + E(z⁺) − E(z⁻))`(`use_margin`/`margin` 可配),一旦分开 margin 就停止推,
   天然避免能量尺度发散;此相位所有 VAE 侧输入(含 embedding)均 detach。
③ **相位 B — 更新 {θ, φ_b, φ_a, embeddings}**:全程冻结能量网络,
   `L = recon + β_b·KL(q_b‖N(0,I)) + β_a·KL(q_a‖N(0,I)) + w·E[E_α] + α_con·L_contrast`。
   **KL 权重取 scGen 量级 5e-5**(刻意很小)。recon 默认 `mse+mmd`。

**推理(论文 Inference)**:未扰动细胞 `x0,c0` → `z_b0` → 采 M 个
`z_a ~ q(z_a|z_b0,a)` → **取能量最低者** → `p_θ(x|z_b0,z_a*)`。

## 数据(与所有既有模型对齐,可直接比较)
Tahoe-100M,5000-HVG,log-norm(`normalize_total(1e4)+log1p`),shard 级
train/val 划分(每第 20 个 shard 为 val),batch 512,AdamW lr 1e-3。
`c` = `cell_line_id`(50),`a` = `drug`(380,含 `DMSO_TF` 对照)。

## 运行
```bash
cd PerturbEnergy
python build_vocab.py                    # 一次:生成 cell_line 词表(drug 词表已存在)

qsub submit_train.sh                     # 训练 + 完整验证(默认 100k 步)
qsub -v STEPS=200000 submit_train.sh     # 改步数
qsub -v OVERRIDES="langevin.steps=40 optim.energy_lr=5e-5" submit_train.sh   # 改超参
qsub submit_eval.sh                      # 只跑验证(已有 checkpoint)
qsub -v TASKS="predict retrieval" submit_eval.sh
```
所有超参在 **`config.yaml`**;命令行可临时覆盖:`--set langevin.steps=40 loss.mmd_weight=5`。
产物 → `/scratch/.../models/perturbenergy/`(`perturbenergy.pt`、`metrics.csv`、
`eval_results.json`、`config_used.json`),**不占项目目录**。

## 验证任务(`evaluate.py --tasks ...`)
| 任务 | 内容 | 可比对象 |
|---|---|---|
| `recon` | held-out 重建 MSE/R²/gene & cell Pearson | 既有 Tahoe VAE(R² 0.49) |
| `predict` | **扰动预测**:DMSO 对照细胞 → 预测该药响应;R²/Pearson 的 mean 与 delta、all genes + top-50 DEG、外加 encode-real 上界 | LDM 结果(`r2_delta_deg` 最好 +0.096) |
| `retrieval` | **能量式药物解码**(Eq.18):对 380 个候选药按 `E_α` 排序,报 top-1/top-5/MRR,含**群体级**聚合版 | DD 探针(单细胞 ~0.04–0.06)、伪批量 DD |
| `probe` | 从 `z_b`/`z_a` 线性&MLP 解码 drug + Energy PP-SNR + 塌缩诊断(within-cond std、effective rank) | 之前的 DD / PP-SNR |

> ⚠️ **`z_a` 上的 drug 探针是循环的**:`z_a` 本身由 `a` 计算而来,所以它的 DD/PP-SNR
> 必然很高,**只能当诊断**,不能作为"隐空间含多少扰动信息"的证据。非循环的检验是
> **`retrieval`**(用能量对候选药排序)以及 `z_b` 上的 DD(应当低 = 解耦良好)。
> 同时监控 `z_a_std` / `effective_rank`:能量项可能把同药细胞压成一点(representation
> collapse),那会虚高 PP-SNR。

## 需要注意的实现选择(可在 config 改)
1. **候选选择** — `n_candidates`(默认 4)、`select_mode`(`min` | `softmax`)、`select_tau` 可调。
2. **`recon: mse+mmd`** — 论文说用 MMD 近似重建项;纯 MMD 没有逐细胞梯度会让 decoder 变差,
   故默认二者并用(`mmd_weight` 可调,设 `recon: mmd` 可复现纯 MMD)。
3. **`z_b_dim=32, z_a_dim=32`** — 与 32 维基线严格同维对比时可设 16/16。
4. **`energy_lr` 比 `vae_lr` 小 10×** + `energy_l2_reg` — EBM 训练稳定性常规做法。
