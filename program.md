# SDF-GAN Autoresearch

An LLM autonomously improves a deep learning asset pricing model. The model learns a Stochastic Discount Factor (SDF) via adversarial training (GAN), following Chen, Pelger & Zhu (2019) — *"Deep Learning in Asset Pricing"*.

## Setup

1. **Read `train.py`** — the only file you modify. Model architecture, optimizer, training loop.
2. **Read the domain brief below** (§ Domain Knowledge) — it contains everything you need from `prepare.py`. Do NOT read `prepare.py` directly; it wastes context.
3. **Initialize results.tsv** with just the header row if it doesn't exist. The baseline will be recorded after the first run.
4. **First run**: establish the baseline by running the training script as-is.

## Goal

**Maximize valid_sharpe by searching across model structure, training dynamics, and information compression.**

The metric hierarchy:
1. **Primary: valid_sharpe** — higher is better
2. **Diagnostic: train_sharpe** — track it. A model with train_sharpe=3.0 and valid_sharpe=1.3 is more interesting than one with train_sharpe=1.6 and valid_sharpe=1.6.

**What makes a good experiment**: a hypothesis-driven change that produces qualitatively different behavior — different train/valid dynamics, different convergence patterns, different information flow. A well-motivated change to phase lengths or LSTM bottleneck width is as valuable as a new layer type.

## Experimentation

Each experiment trains the SDF-GAN end-to-end (3 phases: unconditional → moment update → conditional) on a remote AWS GPU instance. **Never run `python train.py` locally** — there is no local GPU.

**Constraints:**
- Modify only `train.py`. Everything in it is fair game: architecture, optimizer, hyperparameters, training schedule, loss functions, regularization, activation functions.
- Do not modify `prepare.py`, install new packages, or modify the evaluation harness.
- Each run should complete in ~2–30 minutes. If a run exceeds 30 minutes, kill it and treat it as a failure.

## Output format

The script prints a summary block:

```
---
valid_sharpe:     0.452300
train_sharpe:     0.523400
valid_loss:       0.001234
train_loss:       0.000987
valid_ev:         0.031200
train_ev:         0.040100
train_time_s:     120.5
peak_vram_mb:     1234.5
```

## Logging results

Log to `results.tsv` (tab-separated). Do NOT commit results.tsv — leave it untracked.

Header row and 7 columns:

```
commit	category	valid_sharpe	train_sharpe	valid_ev	status	description
```

1. git commit hash (short, 7 chars)
2. category: `structure`, `dynamics`, `compression`, `composition`, `follow-up`, or `baseline`
3. valid_sharpe — use 0.000000 for crashes
4. train_sharpe — use 0.000000 for crashes
5. valid_ev (explained variation) — use 0.000000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description

## The experiment loop

The search operates as a **tree over experiments**. Each kept experiment is a node you can branch from. The TSV is your map of the tree.

### Experiment categories

Each experiment belongs to exactly one category. Label it in the commit message and TSV.

**Structure**: Add, remove, or modify an `nn.Module` subclass — new layer types, conditioning mechanisms, attention, multi-head designs. Changes the computational graph. Requires a 3-sentence hypothesis (see Design budget).

**Dynamics**: Change the training schedule, optimizer, phase structure, learning rate, epoch counts, sub_epoch, or loss formulation. Architecture stays the same; how it's trained changes. Requires a 1-sentence hypothesis.

**Compression**: Change the information bottleneck — LSTM hidden size, RNN type, macro feature processing, or how macro state enters the FFN. Doesn't add new module types, but reshapes what information the model can use. Requires a 1-sentence hypothesis.

**Composition**: Combine elements from two or more kept experiments. Requires listing which keeps are being combined and why.

**Follow-up**: Targeted refinement of a kept or near-miss experiment — tune a hyperparameter, adjust a dimension, fix a clearly suboptimal setting. Requires a 1-sentence description.

Aim for roughly: **40% Structure, 20% Dynamics, 15% Compression, 15% Follow-ups, 10% Compositions.** Do not spend more than 3 consecutive experiments in the same category unless actively ablating a failure.

### Rules

**Ablation**: When an experiment changes more than one thing and fails, the next experiment **must** isolate one of those changes. This applies across all categories. Only after ablation can you move to a new idea.

**Near-miss**: Any experiment with valid_sharpe within 15% of its parent must get at least one Follow-up before being abandoned.

**Borderline confirmation**: When a single run produces valid_sharpe within 10% of its parent (above or below), re-run with 2 additional seeds (`torch.manual_seed` 43 and 44). Use the **median** of the 3 runs for the keep/discard decision. Log the median in results.tsv; note the seed range in the description.

**Keep criteria**: An experiment is **kept** if valid_sharpe improved over its parent. Additionally, keep any experiment within 5% of the global best that has a qualitatively different approach — these "diversity keeps" provide material for compositions.

### Design budget

Before each **Structure** experiment, write a **3-sentence hypothesis** in the commit message:
1. **What** you are changing
2. **Why** it should improve the SDF (connect to loss mechanics or data structure)
3. **What would falsify it** (what result means this direction is wrong, not just needs tuning)

All other categories require a 1-sentence hypothesis or description.

### Node selection

- **Default**: branch from the **global best** valid_sharpe in the TSV.
- **Structure experiments**: branch from baseline unless building on a structural innovation from a prior keep.
- **After 10 consecutive discards**: switch to a different category than the one that produced the last 3 failures.
- **After 20 consecutive discards with no new keep**: write a structured summary, then stop.

To branch from a specific commit: `git show <commit_hash>:train.py > train.py`

### The loop

LOOP FOREVER:

1. Read `notes.md` if it exists
2. Choose a category and parent node
3. If branching from a non-HEAD node: `git show <commit_hash>:train.py > train.py`
4. Modify `train.py`
5. `git add train.py && git commit -m "<category>: hypothesis"`
6. Run on AWS:
   ```bash
   bash aws/sync.sh && bash aws/run-job.sh train && bash aws/download.sh
   ```
   If the instance is unreachable: `bash aws/launch.sh && bash aws/setup.sh`
7. Read results — **only the summary block**: `tail -n 15 aws/results/run.log`
   Do NOT read the full training log.
8. If the summary block is missing: `tail -n 50 aws/results/run.log`. Fix or move on.
9. Log to `results.tsv`
10. Write a **one-sentence "what I learned"** — not what you tried, but what the result tells you about the problem.
11. Apply keep criteria:
    - **Kept**: commit stays. Update global best if applicable.
    - **Discarded**: tag then undo:
      ```bash
      git tag discarded/<short-description> HEAD
      git reset --soft HEAD~1 && git checkout HEAD -- train.py
      ```
      **NEVER use `git reset --hard`.**
12. Check ablation, near-miss, and borderline rules before choosing the next experiment.

### Reflection (every 10 experiments)

Write a structured analysis in `notes.md`:

1. **Per-category summary**: what was tried, what worked, what each result tells you
2. **Cross-category patterns**: do certain categories interact?
3. **What to try next**: which categories and directions look most promising?
4. **Composition candidates**: which keeps have complementary strengths?

### Crashes

Use your judgment: easy fix → fix and re-run. Fundamentally broken → log "crash" and move on.

**Do not pause to ask the human.** The loop runs until the 20-discard halt condition or manual interruption. If you run out of ideas, consult the checklist, re-read notes.md, or try combining ideas.

## Domain knowledge: Asset Pricing with SDF-GAN

### The SDF framework
The Stochastic Discount Factor $M_t = 1 + \sum_i R_{t,i} \cdot w_{t,i}$ prices all assets. The model learns weights $w$ that minimize the moment condition $E[R \cdot M \cdot h] = 0$ for all conditioning instruments $h$.

### Architecture (from the paper)

**Generator:**
1. LSTM(178→4): compresses 178 macro features into 4 hidden states per timestep
2. The 4-dim output is tiled across all N stocks and concatenated with 46 individual features → 50-dim input per stock-month
3. FFN([64,64], ReLU, dropout=0.05): processes each stock independently → 1 scalar weight
4. SDF construction: $M_t = 1 + \sum_i R_{t,i} \cdot w_{t,i}$

The macro conditioning is essential — same stock features should produce different weights in different macro regimes. The paper shows macro states add ~10% OOS Sharpe.

**Adversary:**
- Separate LSTM(178→32): independent macro encoder
- Single linear layer with tanh: 78-dim input → 8 moment conditions
- 0 hidden layers (result of 384-config hyperparameter search)
- Finds the hardest conditioning instruments $h \in [-1, 1]$

**3-phase training (strictly sequential):**
- Phase 1: generator vs unconditional instruments. Checkpoint by best valid loss.
- Phase 2: freeze generator, train adversary. Reload best checkpoint.
- Phase 3: freeze adversary, train generator. Fresh optimizer. Checkpoint by best valid Sharpe.

### Loss mechanics
- `moment_loss`: time-averaged moment conditions `R × SDF × h` per stock, squared, normalized by observation count, averaged across K bases. Gradients flow through SDF.
- `valid_sharpe` = mean/std of monthly portfolio returns where returns = `1.0 - SDF[:, 0]`.
- `evaluate()` chains RNN states across splits: `h_train → h_valid → h_test`. The model must expose `compute_weights_and_sdf(I_macro, I_indiv, R, mask, h0=None)` returning `(w_flat, sdf, rnn_state)`.
- `residual_loss` = explained variation = `1.0 - MSE_residual / MSE_return`. Diagnostic only.
- The paper's in-sample SR=2.68 vs test SR=0.75 — substantial overfitting is expected.

### Data shapes
- **46 individual features** per stock per month, **178 macro features** per month
- **Train**: 240 months, **Valid**: 60 months, **Test**: 300 months
- Variable N_stocks per split (stocks with ≥1 non-UNK observation)
- Macro features normalized per-split using train statistics (no lookahead)
- UNK returns = -99.99, masked out in loss via `loss_weight`

### Key constraints
- 240 training months is small → models must be expressive enough to learn but can't memorize
- The model needs SDF **variance** to generate Sharpe (Sharpe = mean/std of 1-SDF)
- The LSTM output serves two roles: (1) macro conditioning per timestep, and (2) state chaining across splits via h0. Both matter for generalization.
- Each stock is processed independently — cross-sectional interaction is implicit through the SDF portfolio sum.
- The adversary has its own LSTM (32 units) — it can learn different macro representations than the generator

### OOS evaluation

**Do not run `validate.py`.** OOS evaluation is done by the human after the loop ends.

## Unexplored directions

Mark items `[x]` when attempted, add new ideas as they emerge. Use judgement — do not test sequentially.

### Structure
- [x] GELU activation *(kept — global best)*
- [x] FiLM conditioning *(discarded — 0.798)*
- [x] Multi-head SDF *(discarded — 1.254/1.209)*
- [x] Residual FFN with bottleneck *(discarded — 0.897)*
- [x] GLU blocks *(discarded — 1.170)*
- [ ] Self-attention over the 50 input features
- [ ] Temporal attention over macro time series (replace or augment LSTM)
- [ ] Conditional sub-networks: macro state selects which subnetwork processes features
- [ ] Separate time-series vs cross-sectional paths, then combine
- [ ] Deeper adversary: add hidden layer(s) to the moment FFN
- [ ] Feature grouping: process features in semantically meaningful groups
- [ ] Simpler FFN: [64] single layer or [32,32] — test if baseline is overparameterized

### Dynamics
- [ ] Learning rate warmup or cosine annealing
- [ ] Different phase lengths (longer Phase 1, shorter Phase 3, or vice versa)
- [ ] Different sub_epoch counts per phase
- [ ] AdamW with decoupled weight decay
- [ ] Curriculum training: high-volume stocks first
- [ ] Residual loss factor > 0 as auxiliary signal

### Compression
- [ ] LSTM hidden 4 → 8 or 16 (widen the macro bottleneck)
- [ ] GRU instead of LSTM for macro encoder
- [ ] Input noise tuning on macro or individual features
- [ ] Sparse stock weights via L1 or top-k selection

### Composition
- [ ] Stochastic Weight Averaging across top-K checkpoints
- [ ] Ensemble averaging over multiple fits (paper uses 9)
- [ ] Combine best Structure keep with best Dynamics keep
