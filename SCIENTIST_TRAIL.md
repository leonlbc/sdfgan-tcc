# SDF-GAN Experiment Trail

## Starting point

SDF-GAN (Chen, Pelger & Zhu 2019): LSTM(4) compresses 178 macro features, tiles across N stocks, concatenates with 46 individual features, feeds a [64,64] FFN to produce per-stock SDF weights. Adversary (separate LSTM(32) + FFN) finds hardest-to-price test assets. 3-phase training: unconditional, adversary update, conditional. Baseline valid_sharpe: **1.276**.

## What the paper taught us

We extracted and compressed the original paper to identify which design choices were empirically validated vs assumed:

- **No-arbitrage constraint is the regularization.** The GAN structure itself separates signal from noise in the low-SNR return environment. This explains why adding explicit regularization (dropout, L1, weight decay, EMA) on top is neutral-to-harmful.
- **LSTM(4) macro compression is critical.** Raw 178 features cause complete performance collapse. PCA loses dynamics. 4 hidden states optimal (tested 4/8/16/32).
- **Adversary drives 20% of Sharpe gain** via optimal test asset selection, yet our experiments barely touched it.
- **FFN is already sufficient for characteristic interactions.** The paper shows individual characteristics are ~linear; only multi-dimensional interactions need nonlinearity. 2 layers enough.
- **The paper uses 9-model ensemble** to handle initialization variance. We were running single models — the paper's own answer to seed sensitivity was aggregation, not architecture search.
- **Train-to-test decay is worse in the paper (3.6:1)** than our setup (2:1). Our shorter window may help generalization.

This redirected our search: stop modifying the FFN/LSTM, focus on search strategy and output constraints.

## Phase 1: Initial search (experiments 1-33)

### Strategy
Broad search across 5 categories: structure, dynamics, compression, composition, follow-up. Branch from baseline or keeps. 3-seed median protocol for borderline results.

### What happened
33 experiments, 3 keeps (including baseline). The model sat at a sharp local optimum where almost every change degraded performance. Only GELU activation (a drop-in replacement) improved the baseline.

### Key lessons
- **Structure is a dead end**: 9 experiments, 0 keeps. FiLM, multi-head, residual, GLU, deeper adversary, simpler FFN, more moments, macro gate, LayerNorm all hurt. The computational graph is tightly optimized.
- **Training schedule is well-calibrated**: 11 dynamics experiments, 0 keeps. The 256/64/1024 epoch schedule, sub_epoch=4, LR=1e-3 are all near-optimal.
- **Compression bottleneck is critical**: LSTM(8) hurts, GRU hurts, feature projection is catastrophic. The 4-dim bottleneck forces the right compression.
- **Keeps don't compose**: GELU + adversary [32] together scored 1.069 — worse than either alone.
- **Seed variance dominates**: Single-seed results range +/-15% on identical config.

## Phase 2: Novel dimensions (experiments 34-45)

### Strategy
After the paper analysis, we identified 7 untouched dimensions: weight averaging (EMA/SWA), data augmentation (feature noise), warm restarts, asymmetric learning rates, cross-sectional weight constraints, initialization schemes, temporal sub-sampling. Systematically tested each.

### Key discoveries

**Cross-sectional weight centering** (subtract mean weight across stocks per timestep) was the most informative experiment. It creates a long-short constraint forcing relative pricing. Median 1.302, but:
- Train SR exploded to 5.41 (extreme overfitting)
- Seed 43 consistently hit 1.44-1.53 across centering variants
- A much better solution EXISTS but is initialization-dependent

**Temporal dropout** (randomly mask 25% of months in the loss) narrowed the overfitting gap from 2.1:1 to 1.5:1 while maintaining validation performance. Median 1.310.

**Centering + temporal dropout composed**: temporal dropout tamed centering's overfitting (train SR 2.26 vs 5.41). Median 1.318, diversity keep. First successful composition.

**EMA is wrong for this problem**: decay=0.999 caused severe underfitting (1.020). The sharp optimum is the correct solution; smoothing destroys necessary structure.

**Overfitting is temporal, not cross-sectional**: Feature noise (cross-sectional augmentation) didn't help; temporal dropout (temporal augmentation) did.

## Phase 3: Exploiting the centering discovery (experiments 46-53)

### Strategy
The paper's 9-ensemble approach and centering's seed-43 anomaly both pointed to the same insight: the problem is basin-finding reliability, not architecture. We proposed 5 experiments targeting search strategy, game dynamics, and regularization strength.

### Breakthrough

**Multi-restart Phase 3** (4 independent 256-epoch runs with fresh optimizers, keep best checkpoint globally) broke the 1.32 ceiling: median **1.363**. The search strategy was the bottleneck, not the architecture.

**Dropout 0.20 on multi-restart** pushed further to **1.422** — a 7.6% improvement over the old global best. Multi-restart enables higher regularization because if one restart is over-regularized, another finds a better basin. Seed variance 1.279-1.527.

### What failed
- **Soft centering** (alpha 0->1 annealing): catastrophic at 0.788. Gradual transition doesn't work; the constraint must be binary.
- **EMA 0.9995** (lightest possible): still hurts at 1.081. Weight averaging is definitively wrong.
- **Adversary unfreezing** (1 adv step per 10 gen steps in Phase 3): destabilizing at 1.100. The frozen adversary is correct.
- **LR warmup**: hurt convergence at 1.264.
- **Top-5 SWA on multi-restart**: identical to multi-restart alone (1.366 vs 1.363). Checkpoint selection beats checkpoint averaging.

## Keeps

| # | Commit | valid_sharpe | train_sharpe | Description |
|---|--------|-------------|-------------|-------------|
| 1 | 3410e04 | 1.276 | 2.923 | Baseline: [64,64] LSTM(4) ReLU |
| 2 | 4f978bb | 1.322 | 2.796 | GELU activation |
| 3 | f4eeb33 | 1.317 | 1.750 | Adversary [32] hidden (diversity) |
| 4 | c83a7a5 | 1.318 | 2.262 | Weight centering + temporal dropout p=0.75 (diversity) |
| 5 | 8f99735 | 1.363 | 2.778 | Multi-restart Phase 3 (4x256) |
| 6 | 0123174 | 1.422 | 2.237 | Multi-restart + dropout 0.20 |
| 7 | 0ff5230 | 1.484 | 2.131 | Multi-restart + dropout 0.25 |
| 8 | c05c983 | **1.544** | 1.805 | **Multi-restart + dropout 0.30 (global best)** |

## Full experiment log

| # | Commit | Category | valid_sharpe | train_sharpe | valid_ev | Status | Description |
|---|--------|----------|-------------|-------------|----------|--------|-------------|
| 1 | 3410e04 | baseline | 1.276 | 2.923 | 0.002106 | keep | SDF-GAN [64,64] LSTM(4) ReLU |
| 2 | 4f978bb | follow-up | 1.322 | 2.796 | 0.003083 | keep | GELU activation on baseline |
| 3 | 27a5de6 | structure | 0.798 | 3.312 | 0.002629 | discard | FiLM conditioning |
| 4 | b7bf2f4 | structure | 1.254 | 2.105 | 0.002695 | discard | Multi-head SDF K=4 |
| 5 | 4b95536 | follow-up | 1.209 | 2.279 | 0.002711 | discard | Multi-head K=2 simple avg |
| 6 | 5ad1b40 | structure | 0.897 | 2.217 | 0.001973 | discard | Residual FFN bottleneck blocks |
| 7 | d3e0d88 | structure | 1.170 | 2.461 | 0.004241 | discard | GLU blocks in generator |
| 8 | 2163428 | follow-up | 1.045 | 2.690 | 0.003100 | discard | GLU wider 128-dim |
| 9 | e16e013 | structure | 1.194 | 3.018 | 0.003367 | discard | Deeper adversary [64] hidden |
| 10 | 72be420 | compression | 1.028 | 1.935 | 0.003398 | discard | LSTM hidden=8 |
| 11 | f7baec3 | dynamics | 1.178 | 2.264 | 0.003407 | discard | Cosine annealing LR |
| 12 | 175867f | structure | 0.783 | 1.736 | 0.003048 | discard | Simpler [64] single-layer FFN |
| 13 | f4eeb33 | follow-up | 1.317 | 1.750 | 0.002960 | keep | Adversary [32] hidden (diversity) |
| 14 | bbe801c | dynamics | 1.318 | 1.750 | 0.002961 | discard | AdamW weight_decay=1e-4 |
| 15 | 6c4a9e4 | dynamics | 1.160 | 2.365 | 0.003649 | discard | Phase 1=512 |
| 16 | 0abb5ad | follow-up | 1.159 | 1.866 | 0.003070 | discard | Phase 1=384 |
| 17 | b9a78fb | compression | 1.235 | 1.857 | 0.003459 | discard | GRU macro encoder |
| 18 | 6d7188b | dynamics | 0.065 | 0.193 | 0.104025 | discard | Residual loss factor=0.1 |
| 19 | 28f6cea | dynamics | 1.117 | 2.180 | 0.003794 | discard | Gradient clipping 1.0 |
| 20 | 120b534 | dynamics | 1.149 | 2.061 | 0.003753 | discard | LR=5e-4 |
| 21 | a2b9011 | structure | 1.219 | 2.736 | 0.003205 | discard | K=16 moment conditions |
| 22 | 22f4f16 | follow-up | 1.071 | 2.314 | 0.004084 | discard | Dropout=0.05 on adv [32] |
| 23 | b267ee1 | compression | 1.175 | 2.191 | 0.003819 | discard | Adversary LSTM=16 |
| 24 | c2360af | dynamics | 1.281 | 1.707 | 0.003332 | discard | L1 lambda=1e-4 |
| 25 | f53abad | composition | 1.069 | 1.567 | 0.002771 | discard | GELU + adv [32] composition |
| 26 | 26d2dc4 | structure | 0.957 | 3.422 | 0.003817 | discard | Macro-gated feature selection |
| 27 | 57ba0a8 | follow-up | 1.123 | 2.328 | 0.003955 | discard | Adv [32] + L1=1e-5 |
| 28 | d27fb03 | dynamics | 1.097 | 2.293 | 0.003384 | discard | Phase 3=512 epochs |
| 29 | d7e895e | compression | 0.776 | 2.767 | 0.004258 | discard | Feature projection 46->16 |
| 30 | 19848f1 | structure | 0.598 | 2.123 | 0.001678 | discard | LayerNorm on FFN input |
| 31 | 4582b98 | follow-up | 1.277 | 3.313 | 0.003648 | discard | GELU + dropout=0.10 |
| 32 | 8e68646 | dynamics | 1.203 | 1.853 | 0.003055 | discard | sub_epoch=2 |
| 33 | 1aba01a | dynamics | 1.020 | 1.242 | 0.002262 | discard | EMA decay=0.999 |
| 34 | 72fd488 | dynamics | 1.226 | 2.888 | 0.003187 | discard | Feature noise sigma=0.02 |
| 35 | b584948 | dynamics | 1.068 | 2.575 | 0.003832 | discard | Warm restarts T_0=256 |
| 36 | 1cfdcc7 | dynamics | 1.182 | 2.091 | 0.003676 | discard | Asymmetric LR 0.3x LSTM |
| 37 | c00abb3 | follow-up | 1.265 | 1.960 | 0.003051 | discard | Asymmetric LR 0.5x LSTM |
| 38 | cd2df54 | structure | 1.302 | 5.414 | 0.002949 | discard | Cross-sectional weight centering |
| 39 | 4c42971 | follow-up | 1.268 | 5.034 | 0.004248 | discard | Centering + dropout=0.10 |
| 40 | 0bdc5de | dynamics | 1.080 | 1.643 | 0.002824 | discard | Orthogonal LSTM init |
| 41 | 2f1e352 | dynamics | 0.000 | 0.000 | 0.000000 | crash | Temporal dropout lw_train mask |
| 42 | 9e8f2a3 | dynamics | 1.310 | 2.416 | 0.002889 | discard | Temporal dropout p=0.75 |
| 43 | 560ffa2 | follow-up | 1.270 | 2.746 | 0.003059 | discard | Temporal dropout p=0.875 |
| 44 | c83a7a5 | composition | 1.318 | 2.262 | 0.002364 | keep | Centering + temporal dropout p=0.75 |
| 45 | 8f99735 | dynamics | 1.363 | 2.778 | 0.002760 | keep | Multi-restart Phase 3 (4x256) |
| 46 | a45837d | structure | 0.788 | 1.115 | 0.002493 | discard | Soft centering alpha 0->1 |
| 47 | dd1a2e2 | follow-up | 1.319 | 1.816 | 0.003200 | discard | Dropout=0.15 on multi-restart |
| 48 | 501c329 | follow-up | 1.081 | 1.882 | 0.002662 | discard | EMA 0.9995 on multi-restart |
| 49 | 4a428ce | dynamics | 1.264 | 2.144 | 0.002372 | discard | LR warmup 100 epochs |
| 50 | 8329d2c | dynamics | 1.100 | 3.402 | 0.004214 | discard | Adversary unfreezing 1:10 |
| 51 | e1f2ac0 | dynamics | 1.366 | 2.737 | 0.002735 | discard | Top-5 SWA on multi-restart |
| 52 | 0123174 | follow-up | 1.422 | 2.237 | 0.003345 | keep | Dropout=0.20 on multi-restart |
| 53 | 8fe97db | dynamics | 1.374 | 1.943 | 0.003024 | discard | 8x128 restarts — 128 epochs too short |
| 54 | 0ff5230 | follow-up | 1.484 | 2.131 | 0.003166 | keep | Dropout=0.25 — new global best |
| 55 | fbbfd5d | dynamics | 1.001 | 1.777 | 0.003198 | discard | Multi-restart on GELU baseline (NO centering) — 1.001 |
| 56 | a003b97 | composition | 1.083 | 1.955 | 0.003804 | discard | Adversary [32] + centering + multi-restart + dropout=0.25 |
| 57 | c05c983 | follow-up | 1.544 | 1.805 | 0.003401 | keep | **Dropout=0.30 — NEW GLOBAL BEST** |
| 58 | aa8d151 | follow-up | 1.504 | 2.070 | 0.003587 | discard | Dropout=0.35 — curve bending |
| 59 | 4a3b9e5 | follow-up | 1.489 | 1.652 | 0.003435 | discard | Dropout=0.40 — confirms decline past 0.30 |

## Phase 4: Finding the dropout peak (experiments 53-59)

### Strategy
Ride the dropout curve to its peak, plus two critical ablations: (a) does multi-restart work without centering? (b) does the adversary [32] compose with the new regime?

### The dropout curve
| Dropout | valid_sharpe | train_sharpe | Ratio |
|---------|-------------|-------------|-------|
| 0.10 | 1.318 | 2.262 | 1.72:1 |
| 0.15 | 1.319 | 1.816 | 1.38:1 |
| 0.20 | 1.422 | 2.237 | 1.57:1 |
| 0.25 | 1.484 | 2.131 | 1.44:1 |
| **0.30** | **1.544** | **1.805** | **1.17:1** |
| 0.35 | 1.504 | 2.070 | 1.38:1 |
| 0.40 | 1.489 | 1.652 | 1.11:1 |

Peak at dropout=0.30. The train/valid ratio at the peak is 1.17:1 — nearly no overfitting. Beyond 0.30, underfitting sets in (train SR drops to 1.65) while validation also declines.

### Critical ablation: centering is load-bearing
Multi-restart WITHOUT centering scored **1.001** — a complete collapse. Multi-restart alone cannot break the 1.32 ceiling. **Centering is the essential component.** It provides the long-short portfolio constraint that creates a fundamentally better objective. Multi-restart is the search strategy that reliably finds centering's good basins, but centering is what makes those basins exist.

### Adversary [32] still doesn't compose
Adversary [32] + centering + multi-restart + dropout 0.25 scored 1.083. The deeper adversary overwhelms the regularized generator. The standard adversary is correct for this training regime.

### Final configuration (global best, 1.544)
- GELU activation
- Weight centering (w ← w − mean(w) per timestep)
- Temporal dropout p=0.75 (mask 25% of months from loss)
- Multi-restart Phase 3: 4 × 256 epochs, fresh optimizer each
- Dropout 0.30
- Everything else: baseline (LSTM(4), [64,64] FFN, K=8, 256/64/1024 schedule, constant LR=1e-3)

### Improvement trajectory
| Milestone | valid_sharpe | vs baseline | vs paper (1.43) |
|-----------|-------------|------------|-----------------|
| Baseline (ReLU) | 1.276 | — | -10.8% |
| GELU (Phase 1 best) | 1.322 | +3.6% | -7.6% |
| Multi-restart | 1.363 | +6.8% | -4.7% |
| + dropout 0.20 | 1.422 | +11.4% | -0.6% |
| + dropout 0.25 | 1.484 | +16.3% | +3.8% |
| **+ dropout 0.30** | **1.544** | **+21.0%** | **+8.0%** |

## Phase 5: Out-of-Sample Evaluation

Rolling-window OOS: 6 windows × 240-month train / 60-month eval, shifted by 60 months. 360 total OOS months.

### Results

| # | Config | Commit | Valid SR | OOS SR | OOS/Valid | Windows |
|---|--------|--------|---------|--------|-----------|---------|
| 1 | ReLU baseline | 3410e04 | 1.276 | 0.834 | 0.65 | 0.97 / 1.20 / 1.04 / 1.04 / 0.56 / 0.83 |
| 2 | GELU baseline | 4f978bb | 1.322 | **0.964** | **0.73** | 1.23 / 1.58 / 1.71 / 1.00 / 0.43 / 0.65 |
| 3 | Centering + TD | c83a7a5 | 1.318 | 0.799 | 0.61 | 1.21 / 1.24 / 1.06 / 0.83 / 0.63 / 0.55 |
| 4 | Multi-restart d=0.10 | 8f99735 | 1.363 | 0.769 | 0.56 | 1.12 / 1.26 / 1.08 / 0.81 / 0.39 / 0.49 |
| 5 | Dropout 0.30 (best) | c05c983 | 1.544 | **0.961** | 0.62 | 1.34 / 1.53 / 1.24 / 1.27 / 0.86 / 0.62 |

### What the OOS reveals

**The validation "breakthrough" from 1.32 → 1.54 was largely validation-set overfitting.** 59 experiments selected on the same 60 validation months inflated the apparent gains. OOS, the two best models are GELU baseline (0.964) and dropout 0.30 (0.961) — essentially tied.

**GELU is the only change that unambiguously improves OOS.** ReLU 0.834 → GELU 0.964 (+15.6%). A simple activation swap, no added complexity.

**Centering and multi-restart are harmful without sufficient dropout.** Centering + TD alone (0.799) and multi-restart at d=0.10 (0.769) are both WORSE than the ReLU baseline OOS. These components create overfitting that only high dropout can offset.

**Dropout 0.30 recovers but doesn't surpass GELU.** The entire centering + multi-restart + dropout machinery recovers to match the simple GELU model OOS. The 1.544 vs 1.322 validation gap was a mirage.

### Stability analysis

Dropout 0.30 has one clear advantage: **stability across time periods.**

| Window | GELU | Dropout 0.30 | Winner |
|--------|------|-------------|--------|
| 1 | 1.23 | 1.34 | D0.30 |
| 2 | 1.58 | 1.53 | ~tied |
| 3 | 1.71 | 1.24 | GELU |
| 4 | 1.00 | 1.27 | D0.30 |
| 5 | 0.43 | 0.86 | D0.30 |
| 6 | 0.65 | 0.62 | ~tied |

GELU range: 0.43–1.71 (volatile, boosted by one exceptional window). Dropout 0.30 range: 0.62–1.53 (never crashes below 0.62). For a practitioner who cares about worst-case drawdowns, the dropout 0.30 model is preferable despite the similar average.

## Meta-lessons

1. **Validation Sharpe is unreliable after many experiments.** 59 rounds of selection on 60 months inflated apparent gains by ~60% (1.544 valid → 0.961 OOS). The "1.32 ceiling" and "1.54 breakthrough" were artifacts of repeated validation selection. OOS, the real ceiling is ~0.96.
2. **The simplest change was the best.** GELU activation (experiment 2 of 59) produced the best or tied-best OOS result. 57 subsequent experiments of increasing complexity recovered to the same point.
3. **Centering is the breakthrough on validation, a wash on OOS.** The long-short constraint genuinely changes the optimization landscape, but the OOS gains are fully offset by increased overfitting. High dropout compensates exactly — no more, no less.
4. **Search strategy matters, but mostly for validation.** Multi-restart broke the validation ceiling but didn't improve OOS. The paper's 9-ensemble approach works because it averages predictions (reducing variance); our best-checkpoint-selection approach exploits validation instead.
5. **Regularization headroom was real but didn't translate.** Dropout 0.10→0.30 drove validation from 1.318→1.544 but OOS stayed flat. The extra regularization prevented train-time overfitting but the model was already overfitting to the validation window through experiment selection.
6. **Weight averaging is always wrong here.** EMA at 0.999 and 0.9995, SWA over top-5 checkpoints — all hurt or add nothing. The sharp optimum is correct; selection beats averaging.
7. **The paper's methodology was the biggest clue.** Reading the source paper for methodology, not just architecture, was the highest-value research action — even though the resulting multi-restart didn't improve OOS, it correctly identified initialization variance as the key challenge.
8. **Stability is the real contribution of the complex model.** Dropout 0.30 never drops below 0.62 OOS; GELU crashes to 0.43. Same average, better worst case. Whether this justifies the complexity depends on the application.
