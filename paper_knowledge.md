# Chen, Pelger & Zhu (2019) — Condensed Knowledge for Experiments

## Why this architecture

Stock returns have extremely low signal-to-noise. Standard ML forecasts total variation (2nd moment), but most of that is unpredictable news. The no-arbitrage constraint isolates risk premiums (1st moment) — that's where the learnable signal lives. The GAN structure isn't a modeling choice; it's the regularization that makes deep learning work in finance.

## What the authors tried and why each component won

### GAN (adversarial moment selection)
- Problem: infinite possible test assets (moment conditions). Can't evaluate them all.
- Adversary finds assets with largest mispricing, generator fixes them.
- **Optimal test assets → 20% higher Sharpe** vs naive stock-level conditioning.
- Without no-arbitrage, "kitchen-sink" ML fails to beat linear models.

### LSTM(4) macro encoder
- Raw 178 macro features → **performance completely collapses** (worse than no macro at all).
- PCA on differenced macro → misses time-series dynamics (business cycles).
- LSTM recovers hidden states: GAN+LSTM SR=0.64 vs FFN-without-macro SR=0.06 in simulation.
- Tested 4, 8, 16, 32 hidden states. **4 is optimal** — macro has low-dimensional factor structure.

### FFN([64, 64], ReLU, 2 layers)
- Individual characteristics are ~linear in the SDF. **Interactions are nonlinear.**
- 2 layers suffices for interaction complexity. 3-4 layers tested, not superior.
- In simulation: linear SDF collapses to SR=0.01 on two-characteristic interaction; GAN matches population SR=0.94.

### Adversary (separate LSTM(32), D=8 moments)
- D=8 moment instruments → 80,000 effective test assets (D × N stocks).
- Separate LSTM: adversary learns its own macro representation.
- Both D and adversary architecture validated on held-out Sharpe.

### 3-phase training
- Phase 1: unconditional. Phase 2: adversary. Phase 3: conditional.
- **Converges after exactly 3 steps.** Further iterations no improvement.

### 9-model ensemble
- 9 independent fits with different random initializations, averaged.
- Their answer to seed variance. Single-run results are unreliable.

## Paper's quantitative results

- Full test set (1992–2016): SR = 0.75
- Large-cap only (no smallest 40%): SR = 1.73
- Liquid stocks only (no high spread): SR = 2.07
- Validation SR: 1.43
- Train SR: 2.68
- **Train-to-test ratio: 3.6:1** (worse than our 2:1)
- Hyperparameter sensitivity: test SR ranges 0.72–0.77 across top-4 GAN variants
- Rolling window vs full-sample: 0.70 correlation (weakest robustness check)

## What the paper says matters most

1. No-arbitrage constraint IS the regularization — makes deep learning work in low-SNR finance.
2. Macro conditioning requires dynamic models — static/differenced macro is worse than no macro.
3. Test asset selection is critical — adversary drives 20% of the Sharpe gain.
4. Asset pricing is linear in isolation, nonlinear in interaction.

## What the paper never tried

- Cross-sectional interactions between stocks (each stock processed independently)
- Weight constraints/normalization on SDF portfolio
- Data augmentation or temporal resampling
- Alternative SDF functional forms
- Asymmetric learning rates by component
- Weight averaging (SWA/EMA) across training trajectory
- Modifications to the 3-phase schedule (alternating updates, warm restarts)

## Implications for our experiments

### Explains our results
- **All FFN structure changes failed** → paper confirms 2-layer [64,64] is sufficient for interaction complexity. Stop modifying the FFN.
- **Explicit regularization doesn't help** (dropout, L1, weight decay, EMA, feature noise) → the moment conditions already provide the right inductive bias. Adding regularization on top of regularization is redundant.
- **LSTM compression changes are catastrophic** (LSTM=8, GRU, feature proj) → paper confirms 4-dim bottleneck matches the macro data's intrinsic dimensionality.
- **Our 2:1 overfitting ratio is actually good** — paper sees 3.6:1 with the same architecture.

### Reframes the search
- **Ensemble, don't architecture-search.** The paper averages 9 runs to handle initialization variance. Our ±15% seed variance and centering's seed-43 anomaly (1.01→1.53) are exactly what ensembling solves.
- **The adversary is underexplored.** Paper says test asset selection drives 20% of Sharpe. We only changed adversary hidden size. Never changed what the adversary does — its loss, its inputs, its interaction with the generator during Phase 3.
- **Cross-sectional centering is genuinely novel.** Paper processes stocks independently. Our centering result (seed-43 at 1.53) is outside the paper's search space — the one finding that can't be dismissed as "the paper already found the optimum."

### Highest-value next directions
1. **Multi-run aggregation**: ensemble multiple Phase 3 runs (mirroring the paper's 9-fit approach), especially with centering where basin variance is extreme.
2. **Adversary game modifications**: change the adversarial dynamics during Phase 3 (alternating updates, adversary loss modifications, different conditioning), not just adversary size.
3. **Centering stabilization**: the 1.53 solution EXISTS — the problem is reliably reaching it. Soft centering, warmup, multi-restart all attack this.
