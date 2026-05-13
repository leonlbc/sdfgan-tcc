"""
SDF-GAN autoresearch training script. Single-GPU, single-file.
This is the ONLY file the agent modifies.

Usage: python train.py
"""

import os
import sys
import time
import copy
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from prepare import (
    load_data, evaluate, moment_loss, residual_loss, l1_penalty,
    INDIVIDUAL_FEATURE_DIM, MACRO_FEATURE_DIM,
)

# ---------------------------------------------------------------------------
# Configuration — all hyperparameters in one place
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- Feature dimensions (set automatically from data) ---
    individual_feature_dim: int = INDIVIDUAL_FEATURE_DIM
    macro_feature_dim:      int = MACRO_FEATURE_DIM

    # --- Model Layer (generator) architecture ---
    use_rnn:        bool  = True
    rnn_type:       str   = 'LSTM'        # 'LSTM', 'GRU', 'RNN'
    rnn_hidden:     int   = 4
    rnn_layers:     int   = 1
    hidden_dims:    list  = field(default_factory=lambda: [64, 64])
    activation:     str   = 'GELU'         # 'ReLU', 'ELU', 'Tanh', 'LeakyReLU', 'GELU', 'SiLU'

    # --- Moment Layer (adversary) architecture ---
    moment_rnn_type:       str  = 'LSTM'
    moment_rnn_hidden:     int  = 32
    moment_rnn_layers:     int  = 1
    moment_hidden_dims:    list = field(default_factory=list)
    num_moment_conditions: int  = 8

    # --- Regularization ---
    dropout:          float = 0.05
    weight_decay_l2:  float = 0.0
    l1_lambda:        float = 0.0
    batch_norm:       bool  = False
    max_grad_norm:    float = 0.0

    # --- Loss ---
    weighted_loss:        bool  = True
    residual_loss_factor: float = 0.0

    # --- Training schedule ---
    num_epochs_unc:    int   = 256   # Phase 1: unconditional
    num_epochs_moment: int   = 64    # Phase 2: moment update (Chen faithful)
    num_epochs_cond:   int   = 1024  # Phase 3: conditional (total budget across restarts)
    num_restarts:      int   = 4     # Phase 3 multi-restart count
    sub_epoch:         int   = 4     # gradient steps per epoch
    learning_rate:     float = 1e-3
    optimizer:         str   = 'Adam'

    # --- Logging ---
    print_freq:   int = 16
    ignore_epoch: int = 16


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'ELU': nn.ELU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
}


def build_dense_block(in_dim, hidden_dims, out_dim, activation='ReLU',
                      dropout=0.0, batch_norm=False, final_activation=None):
    """Build a Sequential dense network."""
    ActCls = ACTIVATIONS[activation]
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(ActCls())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if final_activation:
        layers.append(ACTIVATIONS[final_activation]())
    return nn.Sequential(*layers)


def init_lstm_forget_bias(lstm_module, value=1.0):
    """Set LSTM forget gate bias to `value` (TF1 default is 1.0)."""
    for name, param in lstm_module.named_parameters():
        if 'bias' in name:
            hidden_size = param.shape[0] // 4
            with torch.no_grad():
                param[hidden_size:2 * hidden_size].fill_(value)


def init_weights_tf_style(module):
    """Initialize weights to match TF1 defaults (Glorot uniform + zero bias)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            if isinstance(m, nn.LSTM):
                init_lstm_forget_bias(m, value=1.0)


# ---------------------------------------------------------------------------
# Model Layer (generator)
# ---------------------------------------------------------------------------

class ModelLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.use_rnn = cfg.use_rnn

        if self.use_rnn:
            RNNCls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[cfg.rnn_type]
            self.rnn = RNNCls(
                input_size=cfg.macro_feature_dim,
                hidden_size=cfg.rnn_hidden,
                num_layers=cfg.rnn_layers,
                batch_first=True,
            )
            self.rnn_input_dropout = nn.Dropout(p=cfg.dropout)
            if cfg.rnn_type == 'LSTM':
                init_lstm_forget_bias(self.rnn, value=1.0)
            macro_out_dim = cfg.rnn_hidden
        else:
            macro_out_dim = cfg.macro_feature_dim

        self.nn = build_dense_block(
            in_dim=cfg.individual_feature_dim + macro_out_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=1,
            activation=cfg.activation,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
        )

    def forward(self, I_macro, I_indiv, mask, h0=None):
        T, N, _ = I_indiv.shape

        if self.use_rnn:
            rnn_in = self.rnn_input_dropout(I_macro.unsqueeze(0))
            rnn_out, rnn_state = self.rnn(rnn_in, h0)
            macro = rnn_out.squeeze(0)
        else:
            macro = I_macro
            rnn_state = None

        macro_tiled = macro.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([I_indiv[mask], macro_tiled[mask]], dim=1)
        w = self.nn(x).squeeze(-1)
        return w, rnn_state


# ---------------------------------------------------------------------------
# Moment Layer (adversary)
# ---------------------------------------------------------------------------

class MomentLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.use_rnn = cfg.use_rnn

        if self.use_rnn:
            RNNCls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[cfg.moment_rnn_type]
            self.rnn = RNNCls(
                input_size=cfg.macro_feature_dim,
                hidden_size=cfg.moment_rnn_hidden,
                num_layers=cfg.moment_rnn_layers,
                batch_first=True,
            )
            self.rnn_input_dropout = nn.Dropout(p=cfg.dropout)
            if cfg.moment_rnn_type == 'LSTM':
                init_lstm_forget_bias(self.rnn, value=1.0)
            macro_out_dim = cfg.moment_rnn_hidden
        else:
            macro_out_dim = cfg.macro_feature_dim

        self.nn = build_dense_block(
            in_dim=cfg.individual_feature_dim + macro_out_dim,
            hidden_dims=cfg.moment_hidden_dims,
            out_dim=cfg.num_moment_conditions,
            activation=cfg.activation,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
            final_activation='Tanh',
        )

    def forward(self, I_macro, I_indiv, mask):
        T, N, _ = I_indiv.shape

        if self.use_rnn:
            rnn_in = self.rnn_input_dropout(I_macro.unsqueeze(0))
            rnn_out, _ = self.rnn(rnn_in)
            macro = rnn_out.squeeze(0)
        else:
            macro = I_macro

        macro_tiled = macro.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([macro_tiled, I_indiv], dim=2)
        h = self.nn(x.reshape(-1, x.shape[-1]))
        h = h.reshape(T, N, -1).permute(2, 0, 1)
        return h


# ---------------------------------------------------------------------------
# Combined SDF-GAN model
# ---------------------------------------------------------------------------

class SDFGAN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model_layer = ModelLayer(cfg)
        self.moment_layer = MomentLayer(cfg)
        init_weights_tf_style(self)

    def compute_weights_and_sdf(self, I_macro, I_indiv, R, mask, h0=None):
        """Forward pass -> per-stock weights and per-period SDF."""
        w_flat, rnn_state = self.model_layer(I_macro, I_indiv, mask, h0=h0)
        R_flat = R[mask]
        wR = R_flat * w_flat

        N_i = mask.sum(dim=1)
        sdf_parts = torch.split(wR, N_i.tolist())
        sdf = torch.stack([p.sum() for p in sdf_parts]).unsqueeze(1) + 1.0
        return w_flat, sdf, rnn_state

    def forward(self, I_macro, I_indiv, R, mask, h0=None):
        w_flat, sdf, rnn_state = self.compute_weights_and_sdf(
            I_macro, I_indiv, R, mask, h0=h0)
        h = self.moment_layer(I_macro, I_indiv, mask)
        return w_flat, sdf, h, rnn_state


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(params, cfg):
    if cfg.optimizer == 'Adam':
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay_l2)
    elif cfg.optimizer == 'AdamW':
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay_l2)
    elif cfg.optimizer == 'SGD':
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay_l2)
    else:
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay_l2)


# ---------------------------------------------------------------------------
# Training loop (3-phase adversarial)
# ---------------------------------------------------------------------------

def train_sdf_gan(cfg, data, seed=42):
    """Full 3-phase training. Returns (model, history_dict)."""
    device = data['device']
    model = SDFGAN(cfg).to(device)
    opt_model = make_optimizer(model.model_layer.parameters(), cfg)
    opt_moment = make_optimizer(model.moment_layer.parameters(), cfg)

    im, ii, r, m = data['train_tensors']
    lw_train = data['lw_train']
    T, N = r.shape

    best_state = None
    best_valid_loss = float('inf')

    def eval_splits():
        res_tr, h_tr = evaluate(model, data['train_tensors'], data['lw_train'])
        res_va, h_va = evaluate(model, data['valid_tensors'], data['lw_valid'], h0=h_tr)
        return res_tr, res_va

    # ============================================================
    # PHASE 1 -- Unconditional
    # ============================================================
    print(f'=== Phase 1: Unconditional ({cfg.num_epochs_unc} epochs) ===')
    t0 = time.time()
    h_ones = torch.ones(1, T, N, device=device)

    for epoch in range(cfg.num_epochs_unc):
        model.train()
        for _ in range(cfg.sub_epoch):
            opt_model.zero_grad()
            w_flat, sdf, _ = model.compute_weights_and_sdf(im, ii, r, m)
            loss = moment_loss(r, m, sdf, h_ones, lw_train)
            if cfg.residual_loss_factor > 0:
                loss = loss + cfg.residual_loss_factor * residual_loss(r, m, w_flat)
            if cfg.l1_lambda > 0:
                loss = loss + cfg.l1_lambda * l1_penalty(w_flat)
            loss.backward()
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.model_layer.parameters(), cfg.max_grad_norm)
            opt_model.step()

        if epoch > cfg.ignore_epoch:
            res_tr, res_va = eval_splits()
            if res_va['loss'] < best_valid_loss:
                best_valid_loss = res_va['loss']
                best_state = copy.deepcopy(model.state_dict())
            if epoch % cfg.print_freq == 0:
                el = time.time() - t0
                print(f'  [UNC] {epoch:4d}/{cfg.num_epochs_unc}  '
                      f'loss {res_tr["loss"]:.4f}/{res_va["loss"]:.4f}  '
                      f'SR {res_tr["sharpe"]:.3f}/{res_va["sharpe"]:.3f}  '
                      f'[{el:.0f}s]')

    print(f'  Phase 1 done in {time.time()-t0:.0f}s. Best valid loss: {best_valid_loss:.6f}')

    # ============================================================
    # PHASE 2 -- Moment update (adversary)
    # Matches TF1 structure: sub_epoch outer loops × num_epochs_moment inner steps.
    # best_moment_loss resets each outer loop (last block wins, matching TF1).
    # ============================================================
    total_moment_steps = cfg.sub_epoch * cfg.num_epochs_moment
    print(f'\n=== Phase 2: Moment update ({total_moment_steps} steps = '
          f'{cfg.sub_epoch} x {cfg.num_epochs_moment}) ===')
    if best_state is not None:
        model.load_state_dict(best_state)

    step = 0
    for sub in range(cfg.sub_epoch):
        best_moment_loss = float('-inf')  # reset each sub_epoch (TF1 parity)
        for inner in range(cfg.num_epochs_moment):
            model.train()
            opt_moment.zero_grad()

            with torch.no_grad():
                w_flat, sdf, _ = model.compute_weights_and_sdf(im, ii, r, m)
            h = model.moment_layer(im, ii, m)
            loss_cond = moment_loss(r, m, sdf.detach(), h, lw_train)

            (-loss_cond).backward()
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.moment_layer.parameters(), cfg.max_grad_norm)
            opt_moment.step()

            lv = loss_cond.item()
            if lv > best_moment_loss:
                best_moment_loss = lv
                best_state = copy.deepcopy(model.state_dict())
            if step % cfg.print_freq == 0:
                print(f'  [MOMENT] {step:4d}/{total_moment_steps}  cond_loss: {lv:.6f}')
            step += 1

    print(f'  Phase 2 done. Best moment loss: {best_moment_loss:.6f}')

    # ============================================================
    # PHASE 3 -- Conditional (multi-restart)
    # ============================================================
    epochs_per_restart = cfg.num_epochs_cond // cfg.num_restarts
    print(f'\n=== Phase 3: Conditional ({cfg.num_restarts} restarts x {epochs_per_restart} epochs) ===')
    phase2_state = copy.deepcopy(best_state) if best_state is not None else None

    t0 = time.time()
    best_valid_sharpe = float('-inf')

    for restart in range(cfg.num_restarts):
        restart_seed = seed + restart * 1000
        torch.manual_seed(restart_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(restart_seed)

        if phase2_state is not None:
            model.load_state_dict(phase2_state)
        opt_model = make_optimizer(model.model_layer.parameters(), cfg)

        print(f'  --- Restart {restart+1}/{cfg.num_restarts} (seed={restart_seed}) ---')

        for epoch in range(epochs_per_restart):
            model.train()
            for _ in range(cfg.sub_epoch):
                opt_model.zero_grad()
                w_flat, sdf, _ = model.compute_weights_and_sdf(im, ii, r, m)
                with torch.no_grad():
                    h = model.moment_layer(im, ii, m)
                loss = moment_loss(r, m, sdf, h, lw_train)
                if cfg.residual_loss_factor > 0:
                    loss = loss + cfg.residual_loss_factor * residual_loss(r, m, w_flat)
                if cfg.l1_lambda > 0:
                    loss = loss + cfg.l1_lambda * l1_penalty(w_flat)
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.model_layer.parameters(), cfg.max_grad_norm)
                opt_model.step()

            if epoch > cfg.ignore_epoch:
                res_tr, res_va = eval_splits()
                if res_va['sharpe'] > best_valid_sharpe:
                    best_valid_sharpe = res_va['sharpe']
                    best_state = copy.deepcopy(model.state_dict())
                if epoch % cfg.print_freq == 0:
                    el = time.time() - t0
                    print(f'  [COND] R{restart+1} {epoch:4d}/{epochs_per_restart}  '
                          f'loss {res_tr["loss"]:.4f}/{res_va["loss"]:.4f}  '
                          f'SR {res_tr["sharpe"]:.3f}/{res_va["sharpe"]:.3f}  '
                          f'[{el:.0f}s]')

        print(f'  --- Restart {restart+1} done. Best valid Sharpe so far: {best_valid_sharpe:.4f} ---')

    print(f'  Phase 3 done in {time.time()-t0:.0f}s. Best valid Sharpe: {best_valid_sharpe:.4f}')

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

cfg = Config()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    t_start = time.time()
    data = load_data(weighted_loss=cfg.weighted_loss)
    model = train_sdf_gan(cfg, data, seed=seed)

    res_tr, h_tr = evaluate(model, data['train_tensors'], data['lw_train'])
    res_va, _ = evaluate(model, data['valid_tensors'], data['lw_valid'], h0=h_tr)

    total_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print("---")
    print(f"valid_sharpe:     {res_va['sharpe']:.6f}")
    print(f"train_sharpe:     {res_tr['sharpe']:.6f}")
    print(f"valid_loss:       {res_va['loss']:.6f}")
    print(f"train_loss:       {res_tr['loss']:.6f}")
    print(f"valid_ev:         {res_va['ev']:.6f}")
    print(f"train_ev:         {res_tr['ev']:.6f}")
    print(f"train_time_s:     {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_mem:.1f}")
