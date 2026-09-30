# CausalDiscForge · 架构与设计文档

> 配套 `README.md`。本文档描述内部数据流、各算法的数学核心、可验证不变量（invariant）与集成策略。

---

## 1. 整体数据流

```
观测数据 X[n,d]
   │
   ├─ discover.pc.pc_stable      ─┐
   ├─ discover.ges.hill_climbing ─┤  单算法族（4 族）
   ├─ discover.notears.notears   ─┤
   ├─ backends.lingam.direct_lingam ┘
   │        │
   │        ▼  每族返回 CPDAG（端点编码矩阵）
   │
   └─ discover.consensus.consensus  ──► 跨 bootstrap × 跨族 共识
            │
            ▼
        CPDAG (graph, edge_freq)
            │
            ▼
   eval.metrics.evaluate  ──► shd_cpdag / skeleton_f1 / orientation_f1 / bic
```

后端（`backends.*`）与自研 4 族输出**同一数据契约**：`CPDAG`（端点编码矩阵 + 可选 `edge_freq`）。所有下游指标、benchmark、CLI 都只消费 `CPDAG`。

---

## 2. CPDAG 数据契约（端点编码）

| 编码 | 值 | 含义 |
|------|----|------|
| `NONE` | 0 | 无边 |
| `TAIL` | -1 | 尾端 `o--`（无向端点） |
| `ARROW` | 1 | 箭头 `-->` |
| `CIRCLE` | 2 | 圆圈 `o-o`（PAG，仅 FCI 后端出现） |

- 有向边 `i → j`：`g[i,j]=ARROW`, `g[j,i]=TAIL`
- 无向边 `i -- j`：`g[i,j]=TAIL`, `g[j,i]=TAIL`
- 对角线恒为 `NONE`（无自环）

`CPDAG.edge_list()` 仅返回 `i<j` 的无序对，`n_edges()` 按 `|g|>0` 计数并 ÷2。

---

## 3. 算法族与数学核心

### 3.1 PC-stable（约束式）
- 条件独立性检验：`Fisher-Z` 偏相关 `r_{ij·S}` → 检验统计量 `√n−|S|−3 · z(r)`。
- `partial_corr` 用**精度矩阵法**（`P = inv(corr)`，`r = −P[i,j]/√(P[i,i]P[j,j])`）数值稳定。
- Skeleton 稳定（重排条件集顺序结果不变），再按 v-结构定向 + Meek R1–R4。
- **不变量**：`pc_oracle`（真值 d-分离）必落在真实 MEC（`shd_cpdag==0`）。

### 3.2 Hill-Climbing / BIC（评分式）
- 分解高斯 BIC：`BIC_i = n·ln(RSS_i/n) + |Pa(i)|·ln n`，全局 BIC 为各节点之和。
- 增量更新：仅受影响节点的局部 BIC 重算。
- **DFS 环检查**：加边 `i→j` 前确认 `j` 不能到达 `i`；反向边 `i→j→j→i` 临时删边后验证 `i` 仍不可达 `j`。
- `best_delta < −1e-8` 才接受操作；`restarts` 随机重启逃局部最优。
- **不变量**：`bic_of_dag` 全局差 == 局部 `_local_bic` 差（增量正确性）。

### 3.3 NOTEARS（连续优化）
- 目标：`min ½n‖X−XW‖²_F + λ‖W‖₁`，约束 `h(W)=tr(e^{W∘W})−d=0`（DAG 充要）。
- 增广拉格朗日 + `L-BFGS-B`；`h` 不达标则增大 `ρ`。
- `_prune_to_dag`：迭代删最小 `|W|` 边直至为 DAG。
- **不变量**：输出邻接必为 DAG（`is_dag` 校验）；`h(W) ≲ 1e-8`。

### 3.4 DirectLiNGAM（非高斯 ICA）
- 非高斯性 → 可识别因果顺序；后端经 `lingam` 提供，缺失则跳过。

---

## 4. 旗舰集成 ConsensusCD（稳定性引导模型选择）

> 早期版本用"边投票"直接拼图，但噪声方法（如非高斯 LiNGAM）的"自信但错误"边会污染
> 稳定性投票，导致集成 SHD 反而劣于单方法。现改为**模型选择**范式。

```
for b in bootstraps:
    子样本 Xb
    for 每族成员 m (pc / hc / notears / +cl_pc / +cl_ges / +lingam?):
        cp = m(Xb)
        present += 边(i,j)
    skel_freq[出现过的边] += 1

skel_frac = skel_freq / bootstraps        # ∈ [0,1]，边置信度（edge_freq）
S*        = (skel_frac >= tau) 且 被 ≥2 个不同算法族确认   # 跨归纳偏置过滤

对全量 X 跑每个成员 -> 候选 CPDAG 集合 C
选择: argmin_{c in C} skeleton_SHD(S*, c)   # 与群体骨架最一致的模型
      tie-break 1: 定向边最少（chain/fork 类无向真值的正确偏好）
      tie-break 2: 最小 BIC
归一端点（CIRCLE->TAIL）保证合法 CPDAG，挂回 edge_freq
```

**设计要点**
- `skel_frac` 按 **bootstrap 出现比例**归一（∈[0,1]），不是按成员数，避免 >1。
- 选择的是**整个方法的 CPDAG**（内部自洽），而非逐边拼装 → 避免不合法/高 SHD 的拼图。
- **刻意不回滚到最低 BIC 基础成员**：早期版本有"选中模型 BIC > 最强基础成员 +2 则回滚"的
  非劣守护，实测反而**放大** CPDAG 误差（平均 SHD 10.67 → 移除后 7.67；BIC 偏好更密/退化
  的图，在 chain/fork 类真值上与 SHD 负相关）。同理跨族确认后的 S* 若用"整体候选与 S* 对齐"
  选择，需注意 S* 由成员自身 bootstrap 而来的循环性——它是无真值下最可靠的一致性信号，
  但在 dense/hard（真值骨架比 S* 更密）上会系统性选偏，属已知限制。
- 该选择是**无真值**的：用"与群体稳定性骨架的一致性"代替未知的真图。实测（真实配置）：
  ConsensusCD 4/6 追平最强单方法、5/6 追平最强基础成员，平均 SHD 5.67（边投票旧版 14.67）。
- LiNGAM 默认不纳入（非高斯范式在线性高斯基准上属错配）；可用 `CAUSALDISC_USE_LINGAM=1` 开启。

---

## 5. SHD 正确口径

`shd_cpdag` 对每对 `i<j` 分类为 `{none, undir, i→j, j→i}`，**仅当两侧类别不同才 +1**。
这正确计入"缺失/多余/反向/部分定向"，而朴素 `‖adj−adj'‖` 会把一条反向边算成 2（双向都差），是错误的。

---

## 6. 配置与可复现

- `Config`：所有可调超参；`ENV_CAUSALDISC_*`（如 `CAUSALDISC_BOOTSTRAPS`）经 `field(default_factory=...)` 在实例化时读取，支持 CI/容器覆盖。
- 锁文件：`requirements.lock.txt`（精确版本，含 `scipy==1.16.3` 满足 lingam 约束）。
- 离线：`numpy/scipy` 即可跑通 4 族核心 + 全部单测；SOTA 后端为可选增强。

---

## 7. 依赖与版本（实测）

| 包 | 版本 | 角色 |
|----|------|------|
| numpy | 2.5.3 | 核心线性代数 |
| scipy | 1.16.3 | 优化 / `expm` (NOTEARS) |
| scikit-learn | 1.9.1 | 工具 |
| causal-learn | 0.1.4.8 | PC / GES / CDNOD / FCI |
| pgmpy | 1.1.2 | Hill-Climbing (`bic-g`) / PC |
| lingam | 1.8.2 | DirectLiNGAM |
| optuna | 5.0.0 | HPO |

> pgmpy ≥1.1 将 `BicScore` 迁移为 `structure_score.BIC`，本仓库用字符串 `"bic-g"` 指定高斯 BIC 评分（连续数据）。

---

© 晨星 (CJX0712)
