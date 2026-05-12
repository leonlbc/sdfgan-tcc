"""
Fixed evaluation harness for SDF-GAN autoresearch.

This file is READ-ONLY — the agent must not modify it.
It contains data loading, loss functions, and evaluation metrics.

The ground truth metric is: valid_sharpe (monthly Sharpe ratio, higher is better).

Usage:
    from prepare import load_data, evaluate_all_splits, print_results
    from prepare import moment_loss, residual_loss, l1_penalty
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

UNK = -99.99
INDIVIDUAL_FEATURE_DIM = 46
MACRO_FEATURE_DIM = 178
N_TRAIN = 240
N_VALID = 60
N_TEST = 300

# Data paths relative to this file's directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "datasets")

CHAR_TRAIN = os.path.join(DATA_DIR, "char", "Char_train.npz")
CHAR_VALID = os.path.join(DATA_DIR, "char", "Char_valid.npz")
CHAR_TEST = os.path.join(DATA_DIR, "char", "Char_test.npz")
MACRO_TRAIN = os.path.join(DATA_DIR, "macro", "macro_train.npz")
MACRO_VALID = os.path.join(DATA_DIR, "macro", "macro_valid.npz")
MACRO_TEST = os.path.join(DATA_DIR, "macro", "macro_test.npz")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class AssetPricingDataset:
    """Loads and holds one split (train/valid/test) of asset pricing data."""

    def __init__(self, path_char, path_macro=None,
                 mean_macro=None, std_macro=None):
        tmp = np.load(path_char)
        data = tmp['data']
        self.R = data[:, :, 0]                         # (T, N)
        self.I = data[:, :, 1:]                        # (T, N, F)
        self.mask = (self.R != UNK)                    # (T, N)
        self.dates = list(tmp['date'])
        self.variables = list(tmp['variable'][1:])     # feature names (skip return)
        self.T, self.N, self.F = self.I.shape

        if path_macro is not None:
            tmp_m = np.load(path_macro)
            self.I_macro = tmp_m['data'].copy()
            self.F_macro = self.I_macro.shape[1]

            if mean_macro is None:
                self.mean_macro = self.I_macro.mean(axis=0)
                self.std_macro = self.I_macro.std(axis=0)
            else:
                self.mean_macro = mean_macro
                self.std_macro = std_macro
            self.I_macro = (self.I_macro - self.mean_macro) / self.std_macro
        else:
            self.I_macro = np.zeros((self.T, 0))
            self.F_macro = 0
            self.mean_macro = None
            self.std_macro = None

    def to_tensors(self, dev):
        return (
            torch.tensor(self.I_macro, dtype=torch.float32, device=dev),
            torch.tensor(self.I, dtype=torch.float32, device=dev),
            torch.tensor(self.R, dtype=torch.float32, device=dev),
            torch.tensor(self.mask, dtype=torch.bool, device=dev),
        )

    def loss_weight(self, dev):
        w = self.mask.sum(axis=0).astype(np.float32)
        return torch.tensor(w, device=dev)


def _check_data_exists():
    """Check that all required data files exist."""
    required = [CHAR_TRAIN, CHAR_VALID, CHAR_TEST,
                MACRO_TRAIN, MACRO_VALID, MACRO_TEST]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing data files:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        print(f"\nPlease place .npz files in: {DATA_DIR}/", file=sys.stderr)
        print("Expected structure:", file=sys.stderr)
        print("  datasets/char/Char_train.npz", file=sys.stderr)
        print("  datasets/char/Char_valid.npz", file=sys.stderr)
        print("  datasets/char/Char_test.npz", file=sys.stderr)
        print("  datasets/macro/macro_train.npz", file=sys.stderr)
        print("  datasets/macro/macro_valid.npz", file=sys.stderr)
        print("  datasets/macro/macro_test.npz", file=sys.stderr)
        sys.exit(1)


def load_data(device=None, weighted_loss=True):
    """Load all three splits. Returns a dict with tensors and metadata.

    Args:
        device: torch device (auto-selects cuda if available)
        weighted_loss: if True, compute per-stock loss weights

    Returns:
        dict with keys:
            train_tensors, valid_tensors, test_tensors: (I_macro, I_indiv, R, mask)
            lw_train, lw_valid, lw_test: loss weights (or None)
            variables: list of feature names
            device: the torch device used
    """
    _check_data_exists()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train = AssetPricingDataset(CHAR_TRAIN, MACRO_TRAIN)
    dl_valid = AssetPricingDataset(CHAR_VALID, MACRO_VALID,
                                   mean_macro=dl_train.mean_macro,
                                   std_macro=dl_train.std_macro)
    dl_test = AssetPricingDataset(CHAR_TEST, MACRO_TEST,
                                  mean_macro=dl_train.mean_macro,
                                  std_macro=dl_train.std_macro)

    print(f"Data loaded:")
    print(f"  Train: T={dl_train.T}, N={dl_train.N}, F_indiv={dl_train.F}, F_macro={dl_train.F_macro}")
    print(f"  Valid: T={dl_valid.T}, N={dl_valid.N}")
    print(f"  Test:  T={dl_test.T},  N={dl_test.N}")
    print(f"  Device: {device}")

    return {
        'train_tensors': dl_train.to_tensors(device),
        'valid_tensors': dl_valid.to_tensors(device),
        'test_tensors': dl_test.to_tensors(device),
        'lw_train': dl_train.loss_weight(device) if weighted_loss else None,
        'lw_valid': dl_valid.loss_weight(device) if weighted_loss else None,
        'lw_test': dl_test.loss_weight(device) if weighted_loss else None,
        'variables': dl_train.variables,
        'device': device,
    }


# ---------------------------------------------------------------------------
# Loss functions (importable by train.py, but ground truth lives here)
# ---------------------------------------------------------------------------


def moment_loss(R, mask, sdf, h, loss_weight=None):
    """Empirical moment condition loss.

    For unconditional: pass h = ones(1, T, N).
    For conditional:   pass h from MomentLayer (K, T, N).

    Loss = mean over K of [ mean over i of (time-avg of R*SDF*h)^2 ]
    """
    T_i = mask.float().sum(dim=0)                          # (N,)
    R_sdf = R * mask.float() * sdf                         # (T, N)
    emp_mean = (R_sdf.unsqueeze(0) * h).sum(dim=1) / T_i   # (K, N)
    sq = emp_mean.pow(2)

    if loss_weight is not None:
        w_norm = loss_weight / loss_weight.max()
        return (sq * w_norm.unsqueeze(0)).mean()
    return sq.mean()


def residual_loss(R, mask, w_flat):
    """Residual pricing error: mean(MSE(R - proj_w R)) / mean(MSE(R))."""
    R_flat = R[mask]
    N_i = mask.sum(dim=1)
    R_parts = torch.split(R_flat, N_i.tolist())
    w_parts = torch.split(w_flat, N_i.tolist())

    res_sq, r_sq = [], []
    for R_t, w_t in zip(R_parts, w_parts):
        coeff = (R_t * w_t).sum() / (w_t * w_t).sum()
        R_hat = coeff * w_t
        res_sq.append((R_t - R_hat).pow(2).mean())
        r_sq.append(R_t.pow(2).mean())
    return torch.stack(res_sq).mean() / torch.stack(r_sq).mean()


def l1_penalty(w_flat):
    """L1 penalty on stock weights for sparsity."""
    return w_flat.abs().mean()


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


def sharpe(r):
    """Monthly Sharpe ratio. r is a numpy array of monthly returns."""
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std())


@torch.no_grad()
def evaluate(model, data_tensors, lw=None, h0=None):
    """Evaluate loss and Sharpe on a single split (no gradients).

    Args:
        model: SDFGAN model (must have compute_weights_and_sdf method)
        data_tensors: (I_macro, I_indiv, R, mask)
        lw: loss weights (or None)
        h0: optional initial RNN hidden state (for chaining across splits)

    Returns:
        (metrics_dict, rnn_state)
    """
    model.eval()
    I_macro, I_indiv, R, mask = data_tensors
    T, N = R.shape

    w_flat, sdf, rnn_state = model.compute_weights_and_sdf(
        I_macro, I_indiv, R, mask, h0=h0)
    h_ones = torch.ones(1, T, N, device=R.device)

    loss = moment_loss(R, mask, sdf, h_ones, lw).item()
    res = residual_loss(R, mask, w_flat).item()
    ev = 1.0 - res  # explained variation

    portfolio = (1.0 - sdf[:, 0]).cpu().numpy()
    sr = sharpe(portfolio)

    model.train()
    return {'loss': loss, 'res_loss': res, 'ev': ev, 'sharpe': float(sr)}, rnn_state


def evaluate_all_splits(model, data):
    """Evaluate on train/valid/test with chained RNN states.

    Args:
        model: trained SDFGAN model
        data: dict from load_data()

    Returns:
        flat dict with all metrics
    """
    res_tr, h_tr = evaluate(model, data['train_tensors'], data['lw_train'])
    res_va, h_va = evaluate(model, data['valid_tensors'], data['lw_valid'], h0=h_tr)
    res_te, _ = evaluate(model, data['test_tensors'], data['lw_test'], h0=h_va)

    results = {}
    for split, res in [('train', res_tr), ('valid', res_va), ('test', res_te)]:
        for k, v in res.items():
            results[f'{split}_{k}'] = v
    return results


def print_results(results):
    """Print results in standardized, grep-friendly format."""
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print("---")
    print(f"valid_sharpe:     {results['valid_sharpe']:.6f}")
    print(f"test_sharpe:      {results['test_sharpe']:.6f}")
    print(f"train_sharpe:     {results['train_sharpe']:.6f}")
    print(f"valid_loss:       {results['valid_loss']:.6f}")
    print(f"test_loss:        {results['test_loss']:.6f}")
    print(f"train_loss:       {results['train_loss']:.6f}")
    print(f"valid_ev:         {results['valid_ev']:.6f}")
    print(f"test_ev:          {results['test_ev']:.6f}")
    print(f"train_ev:         {results['train_ev']:.6f}")
    print(f"peak_vram_mb:     {peak_mem:.1f}")


# ---------------------------------------------------------------------------
# Rolling-window evaluation
# ---------------------------------------------------------------------------


class RollingWindowDataset:
    """Loads all 3 splits once into CPU numpy, assembles rolling windows on demand."""

    def __init__(self):
        _check_data_exists()
        tmp = np.load(CHAR_TRAIN)
        self._char_train = tmp['data']
        tmp = np.load(CHAR_VALID)
        self._char_valid = tmp['data']
        tmp = np.load(CHAR_TEST)
        self._char_test = tmp['data']

        self._macro_train = np.load(MACRO_TRAIN)['data']
        self._macro_valid = np.load(MACRO_VALID)['data']
        self._macro_test = np.load(MACRO_TEST)['data']

        # (char_array, macro_array, global_start, global_end)
        self._splits = [
            (self._char_train, self._macro_train, 0, N_TRAIN),
            (self._char_valid, self._macro_valid, N_TRAIN, N_TRAIN + N_VALID),
            (self._char_test, self._macro_test,
             N_TRAIN + N_VALID, N_TRAIN + N_VALID + N_TEST),
        ]

        print("RollingWindowDataset loaded:")
        print(f"  Train: {self._char_train.shape}, Macro: {self._macro_train.shape}")
        print(f"  Valid: {self._char_valid.shape}, Macro: {self._macro_valid.shape}")
        print(f"  Test:  {self._char_test.shape}, Macro: {self._macro_test.shape}")

    def _assemble_portion(self, global_start, global_end):
        """Extract months [global_start, global_end), pad N to max, filter invalid stocks.

        Returns:
            char_data: (T, N_valid, 47) — only stocks with >=1 valid month
            macro_data: (T, 178)
        """
        char_parts = []
        macro_parts = []
        ns = []

        for char, macro, sp_start, sp_end in self._splits:
            lo = max(global_start, sp_start)
            hi = min(global_end, sp_end)
            if lo >= hi:
                continue
            local_lo = lo - sp_start
            local_hi = hi - sp_start
            char_parts.append(char[local_lo:local_hi])
            macro_parts.append(macro[local_lo:local_hi])
            ns.append(char.shape[1])

        N_max = max(ns)

        padded = []
        for cp, n in zip(char_parts, ns):
            if n < N_max:
                T_s = cp.shape[0]
                pad = np.full((T_s, N_max - n, 47), UNK, dtype=cp.dtype)
                padded.append(np.concatenate([cp, pad], axis=1))
            else:
                padded.append(cp)

        char_data = np.concatenate(padded, axis=0)
        macro_data = np.concatenate(macro_parts, axis=0)

        # Filter out stocks with zero valid months (avoids T_i=0 in moment_loss)
        R = char_data[:, :, 0]
        valid = (R != UNK).any(axis=0)
        char_data = char_data[:, valid]

        return char_data, macro_data

    def get_window(self, global_start, train_size=240, eval_size=60, device=None):
        """Assemble a train+eval window and return tensors.

        Returns dict with:
            train_tensors: (I_macro, I_indiv, R, mask)
            eval_tensors: (I_macro, I_indiv, R, mask)
            lw_train: loss weights for training
            lw_eval: loss weights for eval
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_end = global_start + train_size
        eval_end = train_end + eval_size

        char_tr, macro_tr_raw = self._assemble_portion(global_start, train_end)
        char_ev, macro_ev_raw = self._assemble_portion(train_end, eval_end)

        # Normalize macro using training-portion stats only (no lookahead)
        mean_m = macro_tr_raw.mean(axis=0)
        std_m = macro_tr_raw.std(axis=0)
        std_m[std_m == 0] = 1.0
        macro_tr = (macro_tr_raw - mean_m) / std_m
        macro_ev = (macro_ev_raw - mean_m) / std_m

        def _to_tensors(char, macro, dev):
            R = char[:, :, 0]
            I = char[:, :, 1:]
            mask = (R != UNK)
            return (
                torch.tensor(macro, dtype=torch.float32, device=dev),
                torch.tensor(I, dtype=torch.float32, device=dev),
                torch.tensor(R, dtype=torch.float32, device=dev),
                torch.tensor(mask, dtype=torch.bool, device=dev),
            )

        def _loss_weight(char, dev):
            R = char[:, :, 0]
            mask = (R != UNK)
            w = mask.sum(axis=0).astype(np.float32)
            return torch.tensor(w, device=dev)

        return {
            'train_tensors': _to_tensors(char_tr, macro_tr, device),
            'eval_tensors': _to_tensors(char_ev, macro_ev, device),
            'lw_train': _loss_weight(char_tr, device),
            'lw_eval': _loss_weight(char_ev, device),
        }

    def generate_windows(self, train_size=240, eval_size=60, step=60, device=None):
        """Yield (window_idx, window_data_dict) for all rolling windows."""
        first_eval_start = N_TRAIN
        last_month = N_TRAIN + N_VALID + N_TEST

        window_idx = 0
        eval_start = first_eval_start
        while eval_start + eval_size <= last_month:
            train_start = eval_start - train_size
            window_data = self.get_window(train_start, train_size, eval_size, device)
            yield window_idx, window_data
            eval_start += step
            window_idx += 1


@torch.no_grad()
def evaluate_oos(model, train_tensors, eval_tensors, lw_eval):
    """Run model through training data for RNN state, then evaluate on eval data.

    Returns dict with portfolio_returns (numpy), sharpe, ev, loss.
    """
    model.eval()

    # Forward through training data to obtain RNN hidden state
    I_macro_tr, I_indiv_tr, R_tr, mask_tr = train_tensors
    _, _, rnn_state = model.compute_weights_and_sdf(
        I_macro_tr, I_indiv_tr, R_tr, mask_tr)

    # Evaluate on eval data using the chained RNN state
    I_macro_ev, I_indiv_ev, R_ev, mask_ev = eval_tensors
    T, N = R_ev.shape
    w_flat, sdf, _ = model.compute_weights_and_sdf(
        I_macro_ev, I_indiv_ev, R_ev, mask_ev, h0=rnn_state)

    h_ones = torch.ones(1, T, N, device=R_ev.device)
    loss = moment_loss(R_ev, mask_ev, sdf, h_ones, lw_eval).item()
    res = residual_loss(R_ev, mask_ev, w_flat).item()
    ev = 1.0 - res

    portfolio_returns = (1.0 - sdf[:, 0]).cpu().numpy()
    sr = sharpe(portfolio_returns)

    model.train()
    return {
        'portfolio_returns': portfolio_returns,
        'sharpe': float(sr),
        'ev': ev,
        'loss': loss,
    }


def aggregate_rolling_results(window_results):
    """Concatenate OOS returns across windows, compute overall Sharpe.

    Returns dict with oos_sharpe, oos_ev, per_window_sharpes, n_oos_months.
    """
    all_returns = np.concatenate([wr['portfolio_returns'] for wr in window_results])
    return {
        'oos_sharpe': sharpe(all_returns),
        'oos_ev': float(np.mean([wr['ev'] for wr in window_results])),
        'per_window_sharpes': [wr['sharpe'] for wr in window_results],
        'n_oos_months': len(all_returns),
    }


def print_rolling_results(results, train_sharpes):
    """Print rolling-window results in grep-friendly format."""
    peak_mem = (torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available() else 0)
    print("---")
    print(f"oos_sharpe:       {results['oos_sharpe']:.6f}")
    print(f"avg_train_sharpe: {np.mean(train_sharpes):.6f}")
    print(f"oos_ev:           {results['oos_ev']:.6f}")
    print(f"n_windows:        {len(results['per_window_sharpes'])}")
    print(f"n_oos_months:     {results['n_oos_months']}")
    for i, ws in enumerate(results['per_window_sharpes']):
        print(f"window_{i}_sharpe:  {ws:.6f}")
    print(f"peak_vram_mb:     {peak_mem:.1f}")


# ---------------------------------------------------------------------------
# Main: verify data is loadable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying data setup...")
    _check_data_exists()
    print("All data files found. Ready to train.")
