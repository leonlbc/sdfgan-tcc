# SDF-GAN Autoresearch

This is an experiment to have an LLM autonomously improve a deep learning asset pricing model.

The model learns a Stochastic Discount Factor (SDF) via adversarial training (GAN), following Chen, Pelger & Zhu (2019) — *"Deep Learning in Asset Pricing"*.

## Setup

1. **Read `train.py`** — this is the only file you modify. Model architecture, optimizer, training loop.
2. **Read the domain brief below** (§ Domain Knowledge) — it contains everything you need from `prepare.py`. Do NOT read `prepare.py` directly; it wastes context.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Goal

**Explore architecturally diverse models that achieve high train Sharpe while maintaining generalization.**

The metric hierarchy:
1. **Primary: valid_sharpe** — the main loop metric (higher is better)
2. **Diagnostic: train_sharpe** — track it. High is not bad. A model with train_sharpe=3.0 and valid_sharpe=1.3 is more interesting than one with train_sharpe=1.6 and valid_sharpe=1.6.

**What makes a good experiment**: not a 0.001 improvement from tweaking a learning rate, but a structurally different model that produces qualitatively different behavior — different train/valid dynamics, different weight distributions, different convergence patterns.

## Experimentation

Each experiment trains the SDF-GAN model end-to-end (3 phases: unconditional → moment update → conditional) on a remote AWS GPU instance. **Never run `python train.py` locally** — there is no local GPU. Use the scripts in `aws/` (see The Loop below).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training schedule, loss functions, regularization, activation functions, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**Training time**: Each run should complete in ~2-30 minutes. If a run exceeds 30 minutes, kill it and treat it as a failure.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as is.

## Output format

The script prints a summary:

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

Log to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	valid_sharpe	train_sharpe	valid_ev	status	description
```

1. git commit hash (short, 7 chars)
2. valid_sharpe achieved — use 0.000000 for crashes
3. train_sharpe achieved — use 0.000000 for crashes
4. valid_ev (explained variation) — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

## The experiment loop

The search operates as a **tree over experiments**, not a linear sequence. Each kept experiment is a node you can branch from. The TSV is your map of the tree.

### Operators

**Draft** (primary operator): Change the model's skeleton. New layer types, different conditioning mechanisms, attention, different encoder architectures, multi-head designs, novel training dynamics. See Design budget below for commit message requirements.

**Improve**: Make a targeted change to a kept Draft. Tune it, fix an issue, adjust a hyperparameter that's clearly suboptimal. Use Improve to get the most out of a promising architecture, not to grid-search a boring one.

**Crossover**: Combine structural elements from two different kept architectures.

The ratio should be roughly: **50% Drafts, 40% Improves on promising Drafts, 10% Crossovers.** Explore widely.

### What counts as a Draft (mechanical definition)

A Draft **must add, remove, or structurally modify an `nn.Module` subclass** in `train.py`. If the diff does not touch a class definition that inherits from `nn.Module` (adding a new one, removing one, changing its `forward()` logic or layer composition), it is an Improve, not a Draft — regardless of what the commit message says.

NOT a Draft: changing a hyperparameter, swapping an activation function, adjusting regularization strength, modifying epoch counts, changing optimizer settings. Those are Improves by definition.

### Mandatory ablation rule

When a Draft changes **more than one thing** and fails, your next experiment **must isolate one of those changes**. Do not move on to a completely different idea. Example: if a Draft added FiLM conditioning + residual connections + deeper layers and scored 0.35, the next experiment must test exactly one of {FiLM alone, residuals alone, deeper layers alone} to learn which component caused the failure. Only after ablation can you move to a new idea.

### Near-miss iteration rule

Any experiment with valid_sharpe **within 15% of its parent** must get at least one follow-up Improve before being abandoned. A near-miss (e.g., Draft 5 at 1.097 vs baseline 1.280 = 14% gap) is a signal worth investigating, not a failure to discard and forget. Try a smaller/larger variant, combine it with a known-good technique, or tune its most uncertain hyperparameter.

### Design budget

Before each Draft, write a **3-sentence hypothesis** in the commit message:
1. **What** you are changing structurally
2. **Why** this structure should improve the SDF (connect it to the loss mechanics or data structure)
3. **What would falsify it** (what result would mean this direction is wrong, not just needs tuning)

This is mandatory for Drafts. Improves need only a 1-sentence description.

### Node selection

- **Default**: branch from the **global best** valid_sharpe in the TSV.
- **For Drafts**: always branch from baseline unless the Draft specifically builds on a structural innovation from a prior kept experiment.
- **After 10 consecutive discards**: you must attempt a Draft. Do not continue tweaking the same architecture.
- **After 20 consecutive discards with no new keep**: write a structured summary, then stop.

To start from a specific commit:
```bash
git show <commit_hash>:train.py > train.py
```

### The loop

LOOP FOREVER:

1. Read `notes.md` if it exists — it contains accumulated lessons from prior reflections
2. Choose an operator (Draft / Improve / Crossover) and a parent node
3. If branching from a non-HEAD node: `git show <commit_hash>:train.py > train.py`
4. Modify `train.py`
5. `git add train.py && git commit -m "description of architectural hypothesis"`
6. Run on AWS (produces no stdout — output goes to remote `run.log` only):
   ```bash
   bash aws/sync.sh && bash aws/run-job.sh train && bash aws/download.sh
   ```
   If the instance is unreachable:
   ```bash
   bash aws/launch.sh && bash aws/setup.sh
   ```
7. Read results — **only the summary block**:
   ```bash
   tail -n 15 aws/results/run.log
   ```
   Do NOT read the full training log. Do NOT `cat` or `Read` `run.log`. Intermediate epoch metrics waste context and provide no decision-relevant information. You only need the `---` summary block.
8. If the summary block is missing: crashed. `tail -n 50 aws/results/run.log`. Fix or move on.
9. Log to `results.tsv` (do NOT commit results.tsv — leave it untracked)
10. **Write a one-sentence "what I learned" summary** — not what you tried, but what the result tells you about the problem. Example: "Wider layers increase train_sharpe but valid doesn't follow → capacity alone doesn't help generalization here." This converts raw results into compressed signal for future decisions.
11. Apply keep criteria:
    - **Kept**: commit stays. Update global best if this surpasses it.
    - **Discarded**: undo the experiment commit but keep infrastructure files safe:
      ```bash
      git reset --soft HEAD~1 && git checkout HEAD -- train.py
      ```
      This removes the commit and restores `train.py` to its pre-experiment state without touching other files.
      **NEVER use `git reset --hard`** — it destroys non-experiment files (program.md, scripts).
12. Check the **ablation rule** and **near-miss rule** (above) before choosing your next experiment.

### Keep criteria

An experiment is **kept** if **valid_sharpe improved** over its parent node.

Additionally, keep any experiment that achieves valid_sharpe within 5% of the global best AND has a qualitatively different architecture (different layer types, different connectivity). These "diversity keeps" provide material for crossovers even if they aren't the absolute best.

### Reflection (every 10 experiments)

Pause and write a structured analysis in `notes.md`:

1. **Architectures explored**: what structural ideas were tested? Which showed promise?
2. **Training dynamics**: how did different architectures behave? Different convergence patterns? Different train/valid ratios?
3. **What to try next**: based on what you've seen, what architectural directions look most promising?
4. **Crossover candidates**: which kept architectures have complementary strengths?

### Rolling-window validation

**Do not run `validate.py`.** OOS evaluation is done by the human after the experiment loop ends. This prevents OOS information from entering your context and influencing experiment selection.

### Crashes

If a run crashes, use your judgment: typo or easy fix → fix and re-run. Fundamentally broken → log "crash" and move on. Architectural experiments crash more often than hyperparameter tweaks — that's expected.

**Do not pause to ask the human.** The loop runs until the 20-discard halt condition (see Node selection) or manual interruption. If you run out of ideas, consult the unexplored directions checklist, re-read notes.md, or try combining two ideas that haven't been combined.

## Domain knowledge: Asset Pricing with SDF-GAN

### The SDF framework
The Stochastic Discount Factor $M_t = 1 + \sum_i R_{t,i} \cdot w_{t,i}$ prices all assets. The model learns weights $w$ that minimize the moment condition $E[R \cdot M \cdot h] = 0$ for all conditioning instruments $h$.

### Architecture (from the paper's original code)

**Model Layer (generator):**
1. LSTM(178→4): compresses 178 macro features into 4 hidden state processes per timestep
2. The 4-dim LSTM output is **tiled across all N stocks** at each timestep and **concatenated** with the 46 individual features → 50-dim input per stock-month observation
3. FFN([64,64], ReLU, dropout=0.05): processes the 50-dim input independently per stock → 1 scalar weight per stock
4. SDF construction: $M_t = 1 + \sum_i R_{t,i} \cdot w_{t,i}$

The macro conditioning is essential — the same stock features should produce different weights in different macro regimes (recession vs expansion). The paper shows macro states add ~10% OOS Sharpe.

**Moment Layer (adversary):**
- Separate LSTM(178→32): its own macro encoder, independent from the generator's LSTM
- Single linear layer with tanh: (46 individual + 32 macro) = 78-dim input → 8 moment conditions
- 0 hidden layers — this was the result of searching 384 hyperparameter configs
- The adversary finds the hardest conditioning instruments $h \in [-1, 1]$

**3-phase training (strictly sequential, no alternating):**
- Phase 1: train generator against unconditional (constant) instruments. Checkpoint by best valid loss.
- Phase 2: freeze generator, train adversary to find hardest tests. Reload best-loss checkpoint first.
- Phase 3: freeze adversary, train generator against the harder tests. Reload best checkpoint, fresh optimizer. Checkpoint by best valid Sharpe.
- The paper explicitly states convergence in 3 steps due to averaging over ~80,000 instrumented assets.

### How the loss works (from prepare.py)
- `moment_loss` computes time-averaged moment conditions `R × SDF × h` per stock, squares them, normalizes by each stock's valid observation count `T_i`, then averages across all K moment bases. Gradients flow through SDF to enforce pricing equilibrium.
- `valid_sharpe` = mean / std of monthly portfolio returns, where portfolio returns = `1.0 - SDF[:, 0]`. Higher is better. This is the checkpoint selection criterion in Phase 3.
- `evaluate()` chains RNN hidden states across splits: `h_train → h_valid → h_test`. The model must expose `compute_weights_and_sdf(I_macro, I_indiv, R, mask, h0=None)` returning `(w_flat, sdf, rnn_state)`.
- `residual_loss` = explained variation = `1.0 - MSE_residual / MSE_return`. Diagnostic only.
- The paper's in-sample train SR=2.68 vs test SR=0.75 — substantial overfitting is expected. The relevant metric is relative OOS performance between models.

### Data shapes
- **46 individual features** per stock per month, **178 macro features** per month
- **Train**: 240 months, **Valid**: 60 months, **Test**: 300 months
- Variable N_stocks per split (stocks with ≥1 non-UNK observation)
- Macro features normalized per-split using train statistics (no lookahead)
- UNK returns = -99.99, masked out in loss via `loss_weight`

### Architectural implications
- 240 months of training is small → architectures must be expressive enough to learn but can't be so large they memorize
- valid_sharpe is noisy with only 60 months → changes < 0.01 may be noise
- The model needs SDF **variance** to generate Sharpe (Sharpe = mean/std of 1-SDF)
- The 178 macro features are compressed to 4 states via the LSTM
- The LSTM output serves **two roles**: (1) conditioning the FFN with macro state each timestep, and (2) state chaining across train/valid/test splits via h0. Both matter for generalization.
- The FFN input is 50-dim (46 individual + 4 macro states). Each stock is processed independently — the same shared-weight network maps (features, macro state) → weight. Cross-sectional interaction is implicit through the SDF portfolio sum.
- The adversary has its own separate LSTM (32 hidden units) — it can learn different macro representations than the generator

## Unexplored directions checklist

Modify this checklist as you go. Mark items `[x]` when attempted, add new ideas as they emerge. If you find yourself out of ideas, consult this list — do not repeat directions already tried. Use your judgement and taste instead of testing every direction sequentially.

- [ ] GELU activation
- [ ] Input noise tuning
- [ ] Self-attention over the 50 input features (individual + macro states)
- [ ] Temporal attention over macro time series (replace or augment LSTM)
- [ ] Multi-head SDF with separate aggregation (not gating — averaging, learned weighting)
- [ ] Conditional sub-networks: macro state selects which subnetwork processes features
- [ ] Separate time-series vs cross-sectional processing paths, then combine
- [ ] Curriculum training: train on high-volume stocks first, then include harder ones
- [ ] Deeper adversary: add hidden layer(s) to the moment FFN
- [ ] Feature grouping: process features in semantically meaningful groups before combining
- [ ] Sparse stock weights via L1 or top-k selection
- [ ] Multiplicative macro interaction (FiLM/gating) instead of concatenation
- [ ] Stochastic Weight Averaging across top-K checkpoints
- [ ] Ensemble averaging over multiple fits (paper uses 9)
