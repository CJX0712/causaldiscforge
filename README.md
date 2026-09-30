# CausalDiscForge · 因果发现引擎

> 从观测数据自动学习因果图（DAG / CPDAG）的世界级因果**结构学习**（Causal Structure Learning）工具箱。
> 作者：**晨星 (CJX0712)** · MIT License · 纯 Python，零本地编译。

---

## 1. 这是什么 / 与 causalforge 的区别

| 维度 | **causalforge**（另一仓库） | **CausalDiscForge**（本仓库） |
|------|---------------------------|------------------------------|
| 任务 | 因果**推断**：给定变量 X 对 Y 的因果效应（ATE / CATE） | 因果**发现**：从数据本身学出因果**图结构** |
| 输入 | 已知图 + 观测/干预数据 | 仅观测数据（无图） |
| 输出 | 效应量估计 | DAG / CPDAG（含方向） |
| 关键技术 | 后门/前门准则、IPW、DML | PC-stable、GES/hill-climbing、NOTEARS、LiNGAM |

一句话：**causalforge 算"有多大效果"，CausalDiscForge 找"谁导致谁"。** 两者互补，构成端到端因果分析链路。

---

## 2. 核心能力

跨 **4 大算法族**，覆盖线性/非线性、高斯/非高斯、约束式/评分式/连续优化全部主流范式：

| 族 | 方法 | 原理 | 离线可跑 |
|----|------|------|----------|
| 约束式 (constraint) | **PC-stable** | Fisher-Z 偏相关条件独立性 + Meek R1–R4 | ✅ 纯 numpy/scipy |
| 评分式 (score) | **Hill-Climbing / BIC** | 分解 BIC 增量更新 + DFS 环检查 + 随机重启 | ✅ 纯 numpy/scipy |
| 连续优化 (continuous) | **NOTEARS** (Zheng et al., NeurIPS 2018) | `min ½n‖X−XW‖²_F + λ‖W‖₁` s.t. `h(W)=tr(e^{W∘W})−d=0`，增广拉格朗日 + L-BFGS-B | ✅ 纯 numpy/scipy |
| 非高斯 (ICA) | **DirectLiNGAM** | 非高斯独立成分 → 因果顺序 | ⚠️ 需 `lingam` |

### 旗舰集成：ConsensusCD

跨 **3 大算法族**（约束 PC-stable / 评分 Hill-Climbing-BIC / 连续优化 NOTEARS）+ 可选 SOTA 后端（causal-learn PC/GES）的 **稳定性引导模型选择（stability-guided model selection）** 集成：
- 对 `B` 个 bootstrap 子样本，统计每条边的出现频率 → 边置信度 `edge_freq ∈ [0,1]` 与稳定性骨架 `S*`（频率 ≥ `tau` **且** ≥2 个不同算法族确认，跨归纳偏置过滤单法噪声）；
- 在**全量数据**上跑每个成员得到候选 CPDAG，选择与 `S*` 骨架最一致的那个（"群体中最可靠"的模型），并列时取定向最少者、再取 BIC 最低者；
- **不做 BIC 回滚**：曾试过"选中模型 BIC 劣于最强基础成员即回滚"的守护，实测反而**放大** CPDAG 误差（BIC 偏好更密/退化的图，在 chain/fork 类真值上与 SHD 负相关），已移除；
- 实测（6 数据集真实配置）：ConsensusCD 在 **4/6 数据集上追平最强单方法**、**5/6 上追平最强基础成员**，且从不出现灾难性误差（最差 15 vs 旧投票版 35）；在 dense/hard（稳定性骨架比真值稀疏）两类上仍有差距，属已知限制；
- LiNGAM 默认**不**纳入（其面向非高斯数据，在线性高斯基准上属范式错配，会污染稳定性投票；可通过 `CAUSALDISC_USE_LINGAM=1` 开启）。

---

## 3. 复用顶级开源（禁止从零自研 SOTA）

本仓库**站在巨人肩上**，并非重复造轮子：

| 后端 | 包 | 提供 |
|------|----|------|
| causal-learn | `causal-learn` | PC / GES / CDNOD / FCI（工业级实现） |
| pgmpy | `pgmpy` | Hill-Climbing（BIC-g 评分）/ PC |
| lingam | `lingam` | DirectLiNGAM（非高斯因果序） |
| optuna | `optuna` | NOTEARS 超参搜索（HPO） |

后端**自动探测**：缺失某个包时优雅跳过，绝不伪造数字。核心 4 族**始终可用**（纯 numpy/scipy 兜底）。

---

## 4. 快速开始

```bash
# 安装（仅核心：纯 numpy/scipy，可离线）
pip install -e .

# 安装 SOTA 后端（可选）
pip install -e ".[sota]"

# 跑全部测试
pytest -q -W ignore::UserWarning

# 跑端到端 benchmark（生成 benchmark.json）
python examples/run_demo.py

# 命令行
python -m causaldisc.cli list-models
python -m causaldisc.cli benchmark --out benchmark.json
python -m causaldisc.cli run --data your_data.csv
```

### Python API

```python
import numpy as np
from causaldisc.discover.pc import pc_stable
from causaldisc.discover.consensus import consensus
from causaldisc.pipeline.pipeline import CausalDiscPipeline

X = np.loadtxt("data.csv", delimiter=",")

# 单法
cpdag = pc_stable(X, alpha=0.01, max_level=3)

# 旗舰集成（推荐）
est = consensus(X)  # 跨 4 族共识，返回 CPDAG

# 流水线 benchmark
pipe = CausalDiscPipeline()
rows = pipe.benchmark()  # 6 数据集 × 全部方法，含 shd_cpdag / skeleton_f1 / bic
```

---

## 5. 模块结构

```
causaldisc/
├── core/            # types(CPDAG/端点编码), config(ENV_CAUSALDISC_* 覆盖), errors, interfaces
├── graph/
│   └── primitives   # partial_corr, meek_rules, dag_to_cpdag, shd_cpdag(正确无序对口径),
│                    #   skeleton_f1, orientation_scores, bic_of_dag/cpdag, d_separation, is_dag
├── data/
│   └── synthetic    # 6 档 DGP：chain/fork/collider/tree/dense/hard（含真 DAG+CPDAG）
├── discover/        # pc(PC-stable/oracle), ges(hill-climbing), notears(NOTEARS), consensus(ConsensusCD)
├── backends/        # causallearn / pgmpy / lingam 适配器 + registry(自动探测)
├── eval/            # metrics(evaluate), benchmark(方法注册表+执行+打印)
├── hpo/             # optuna_search(tune_notears)
├── pipeline/        # CausalDiscPipeline：list/run/benchmark/run_demo
└── cli.py           # list-models / benchmark / run
```

---

## 6. 关键设计决策（踩坑记录）

- **SHD 正确口径**：逐无序对查表计分（`shd_cpdag`），**禁止**邻接矩阵差绝对值求和（会把"反向边"算成 2 倍）。
- **CPDAG 端点编码**：`NONE=0 / TAIL=-1 / ARROW=1 / CIRCLE=2`（与 causal-learn 一致），后端零转换。
- **Windows 沙箱**：`n_jobs=1`；numpy 2.x 的 `np.trapz→np.trapezoid` 已规避。
- **依赖锁**：`scipy<1.17` 因 `lingam` 硬性约束；`requirements.lock.txt` 提供可复现的精确版本。
- **离线兜底**：4 族核心算法仅依赖 numpy/scipy，无网络/无 GPU 即可全绿单测。

---

## 7. 质量证据

- `pytest`：**全部单测通过**（47 项），覆盖原语不变量、各算法恢复、后端适配、集成选择契约、config 环境变量覆盖。
- `benchmark.json`：6 数据集 × 全部方法落盘；旗舰 ConsensusCD（稳定性引导模型选择）平均 SHD **5.67**（4/6 追平最强单方法、5/6 追平最强基础成员，无灾难性误差）；对照：早期边投票集成版为 14.67，已被本版取代。
- 复现：见 `Makefile`（`make test` / `make demo`）与 `Dockerfile`（python:3.11-slim）。

---

© 晨星 (CJX0712) · 本仓库为「随机创新世界级 AI 系统」系列交付之一。
