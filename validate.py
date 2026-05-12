"""
Rolling-window robustness evaluation for SDF-GAN.

Human-triggered — not part of the agent experiment loop.
Re-estimates the model on 6 rolling 240-month windows (shifted by 60 months)
and evaluates out-of-sample on the next 60 months, producing 360 months of
OOS data for robust Sharpe estimation.

Uses whatever Config the agent has settled on in train.py.

Usage: python validate.py
"""

import time
import copy

import torch
import torch.nn as nn

from prepare import (
    evaluate, evaluate_oos,
    moment_loss, residual_loss, l1_penalty,
    RollingWindowDataset, aggregate_rolling_results, print_rolling_results,
)
from train import SDFGAN, make_optimizer, cfg


# ---------------------------------------------------------------------------
# Rolling-window training (one window)
# ---------------------------------------------------------------------------

WARM_EPOCHS_UNC = 32
WARM_EPOCHS_COND = 192


def train_window(cfg, train_tensors, lw_train, device,
                 init_state_dict=None, is_first_window=True):
    """Train one rolling window. 3-phase, best-state tracked on training metrics."""
    model = SDFGAN(cfg).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    opt_model = make_optimizer(model.model_layer.parameters(), cfg)
    opt_moment = make_optimizer(model.moment_layer.parameters(), cfg)

    im, ii, r, m = train_tensors
    T, N = r.shape

    best_state = None
    best_train_loss = float('inf')

    epochs_unc = cfg.num_epochs_unc if is_first_window else WARM_EPOCHS_UNC
    epochs_cond = cfg.num_epochs_cond if is_first_window else WARM_EPOCHS_COND

    # ============================================================
    # PHASE 1 -- Unconditional
    # ============================================================
    print(f'  === Phase 1: Unconditional ({epochs_unc} epochs) ===')
    t0 = time.time()
    h_ones = torch.ones(1, T, N, device=device)

    for epoch in range(epochs_unc):
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
            res_tr, _ = evaluate(model, train_tensors, lw_train)
            if res_tr['loss'] < best_train_loss:
                best_train_loss = res_tr['loss']
                best_state = copy.deepcopy(model.state_dict())
            if epoch % cfg.print_freq == 0:
                el = time.time() - t0
                print(f'    [UNC] {epoch:4d}/{epochs_unc}  '
                      f'loss {res_tr["loss"]:.4f}  SR {res_tr["sharpe"]:.3f}  [{el:.0f}s]')

    print(f'  Phase 1 done in {time.time()-t0:.0f}s.')

    # ============================================================
    # PHASE 2 -- Moment update
    # ============================================================
    total_moment_steps = cfg.sub_epoch * cfg.num_epochs_moment
    print(f'  === Phase 2: Moment update ({total_moment_steps} steps) ===')
    if best_state is not None:
        model.load_state_dict(best_state)

    step = 0
    for sub in range(cfg.sub_epoch):
        best_moment_loss = float('-inf')
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
            step += 1

    print(f'  Phase 2 done.')

    # ============================================================
    # PHASE 3 -- Conditional
    # ============================================================
    print(f'  === Phase 3: Conditional ({epochs_cond} epochs) ===')
    if best_state is not None:
        model.load_state_dict(best_state)

    opt_model = make_optimizer(model.model_layer.parameters(), cfg)
    t0 = time.time()
    best_train_sharpe = float('-inf')

    for epoch in range(epochs_cond):
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
            res_tr, _ = evaluate(model, train_tensors, lw_train)
            if res_tr['sharpe'] > best_train_sharpe:
                best_train_sharpe = res_tr['sharpe']
                best_state = copy.deepcopy(model.state_dict())
            if epoch % cfg.print_freq == 0:
                el = time.time() - t0
                print(f'    [COND] {epoch:4d}/{epochs_cond}  '
                      f'loss {res_tr["loss"]:.4f}  SR {res_tr["sharpe"]:.3f}  [{el:.0f}s]')

    print(f'  Phase 3 done in {time.time()-t0:.0f}s. Best train Sharpe: {best_train_sharpe:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_train_sharpe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(f"seed:             {args.seed}")

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = RollingWindowDataset()

    window_results = []
    train_sharpes = []
    prev_state_dict = None
    step = 60

    for win_idx, window_data in dataset.generate_windows(step=step, device=device):
        eval_start = 240 + win_idx * step
        train_start = eval_start - 240
        print(f'\n{"="*60}')
        print(f'Window {win_idx}: train [{train_start}-{train_start+239}], '
              f'eval [{eval_start}-{eval_start+59}]')
        T_tr = window_data['train_tensors'][2].shape[0]
        N_tr = window_data['train_tensors'][2].shape[1]
        T_ev = window_data['eval_tensors'][2].shape[0]
        N_ev = window_data['eval_tensors'][2].shape[1]
        print(f'  Train: T={T_tr}, N={N_tr}  |  Eval: T={T_ev}, N={N_ev}')
        print(f'{"="*60}')

        is_first = (win_idx == 0)
        init_state = prev_state_dict if not is_first else None

        model, train_sharpe = train_window(
            cfg, window_data['train_tensors'], window_data['lw_train'],
            device, init_state_dict=init_state, is_first_window=is_first,
        )
        train_sharpes.append(train_sharpe)

        oos = evaluate_oos(
            model, window_data['train_tensors'],
            window_data['eval_tensors'], window_data['lw_eval'],
        )
        window_results.append(oos)
        print(f'  OOS Sharpe: {oos["sharpe"]:.4f}, OOS EV: {oos["ev"]:.4f}')

        prev_state_dict = copy.deepcopy(model.state_dict())

        del model
        torch.cuda.empty_cache()

    results = aggregate_rolling_results(window_results)
    print_rolling_results(results, train_sharpes)
    print(f"\ntrain_time_s:     {time.time() - t_start:.1f}")
