import os
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
from utils.load_models import load_d

def gamma(t, p, q):
    # p, q shape: nmodels, nsamples, nclasses
    cospq = np.clip((p*q).sum(-1), 0, 1)
    ti = np.arccos(cospq)
    mask = ti < 1e-8
    gamma = np.zeros_like(p)
    gamma[mask, :] = p[mask, :]
    p, q = p[~mask, :], q[~mask, :]
    ti = ti[~mask, None]
    gamma[~mask, :] = np.sin((1-t)*ti) / np.sin(ti) * \
        p + np.sin(t*ti) / np.sin(ti) * q
    return gamma

def project(r, p, q, debug=False, mode='mean'):
    # r, p, q: (nm, ns, nc), unit-length over classes (sqrt-probs)
    # Returns lam: (nm,) by averaging per-sample projection parameters
    nm, ns, nc = r.shape
    # Precompute per-sample plane geometry (same across nm because p,q are repeated)
    cospq = np.clip((p * q).sum(-1), 0.0, 1.0)               # (nm, ns)
    alpha = np.arccos(cospq)                                  # (nm, ns)
    s = np.sin(alpha)                                         # (nm, ns)
    # Build orthonormal u direction for each sample
    # u = (q - cospq * p) / sin(alpha)
    u = q - cospq[..., None] * p                              # (nm, ns, nc)
    mask = s > 1e-8
    u[mask] /= s[mask][..., None]
    # For degenerate cases (shouldn’t happen with uniform vs one-hot), fall back to p
    u[~mask] = p[~mask]

    # Per-model/time t via per-sample angle formula, then mean across samples
    dot_rp = (r * p).sum(-1)                                  # (nm, ns)
    dot_ru = (r * u).sum(-1)                                  # (nm, ns)
    t = np.arctan2(dot_ru, dot_rp)                            # (nm, ns), in radians of alpha
    # Avoid division by zero
    alpha_safe = np.where(alpha < 1e-8, 1.0, alpha)
    lam = (t / alpha_safe).clip(0.0, 1.0)                     # (nm, ns)
    # Aggregate across samples
    lam_out = lam.mean(axis=1)                                # (nm,)
    return lam_out

def compute_lambda(file_list, reparam=False, force=False, 
                   didx_loc='inpca_results_all/corners', align_didx='',
                   didx_fn='all', save_loc='results/models/reindexed_new',
                   labels_override=None, nclasses=None):
    """
    labels_override:
      - dict {'train': y_train, 'val': y_val}: use these labels for all files
      - callable(path)->dict: per-file labels (load from saved run)
      - None: Not supported in this minimal impl (avoids get_data). Supply labels_override.
    nclasses: override number of classes
    """
    os.makedirs(didx_loc, exist_ok=True)
    os.makedirs(save_loc, exist_ok=True)

    def _mk_qp(y):
        y = np.asarray(y, dtype=np.int32)
        K = int(nclasses) if nclasses is not None else int(y.max() + 1)
        onehot = np.zeros((y.size, K), dtype=np.float32)
        onehot[np.arange(y.size), y] = 1.0
        q = np.sqrt(onehot)[None, ...]              # (1, N, K)
        p = np.sqrt(np.ones_like(q) / K)            # (1, N, K)
        return q, p, y

    # Prepare global q/p/labels if provided as dict
    global_qs = {}
    global_ps = {}
    global_labels = {}
    if isinstance(labels_override, dict):
        q_tr, p_tr, y_tr = _mk_qp(labels_override['train'])
        q_va, p_va, y_va = _mk_qp(labels_override['val'])
        global_qs = {'yh': q_tr, 'yvh': q_va}
        global_ps = {'yh': p_tr,  'yvh': p_va}
        global_labels = {'yh': y_tr, 'yvh': y_va}
    elif labels_override is None:
        raise ValueError("compute_lambda: labels_override is required to avoid get_data().")

    didx_all = None
    cols = ['seed', 'iseed', 'isinit', 'corner', 'aug', 'm', 'opt', 'bs', 'lr', 'wd',
            'bsel', 't', 'err', 'verr', 'lam_yh', 'lam_yvh']

    def _needs_recompute(df) -> bool:
        try:
            missing = [c for c in cols if c not in df.columns]
            return len(missing) > 0
        except Exception:
            return True

    for f in tqdm(file_list, desc="compute_lambda"):
        save_fn = os.path.join(save_loc, os.path.basename(f))

        d = None
        # Fast path: if we already have a reindexed dataframe, use it to build didx.
        if os.path.exists(save_fn) and not force:
            try:
                d = th.load(save_fn, weights_only=False)
            except Exception as e:
                print(f"Failed to load existing reindexed {save_fn}: {e}")
                d = None

        # If missing or incompatible, load from original run and (re)compute.
        if d is None or _needs_recompute(d):
            try:
                d = load_d(file_list=[f], avg_err=True, probs=False)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                continue
            if d is None or len(d) == 0:
                print(f"Empty run: {f}")
                continue

            # Use per-file labels if a callable is provided; else use global
            if callable(labels_override):
                try:
                    lab = labels_override(f)  # expects {'train': ..., 'val': ...}
                    q_tr, p_tr, y_tr = _mk_qp(lab['train'])
                    q_va, p_va, y_va = _mk_qp(lab['val'])
                    qs = {'yh': q_tr, 'yvh': q_va}
                    ps = {'yh': p_tr, 'yvh': p_va}
                except Exception as e:
                    print(f'Failed to load labels for {f}: {e}')
                    qs, ps = global_qs, global_ps
            else:
                qs, ps = global_qs, global_ps

            # Compute lambda for train/val predictions
            for key in ['yh', 'yvh']:
                yk = np.stack(d[key].values)   # (T, N, K)
                if not np.allclose(yk.sum(-1), 1):
                    yk = np.exp(yk)            # convert from log-probs if needed
                yk = np.sqrt(yk)               # (T, N, K)
                qs_ = np.repeat(qs[key], yk.shape[0], axis=0)
                ps_ = np.repeat(ps[key], yk.shape[0], axis=0)
                lam = project(yk, ps_, qs_)    # (T,)
                d[f'lam_{key}'] = lam

            # Save the per-run dataframe so downstream steps can load it without
            # needing access to the original run format.
            try:
                th.save(d, save_fn)
            except Exception as e:
                print(f"Failed to save reindexed {save_fn}: {e}")

        # Accumulate minimal table (even if we didn't recompute)
        try:
            dd = d.reindex(cols, axis=1)
        except Exception:
            # Best-effort: keep only columns that exist
            dd = d[[c for c in cols if c in getattr(d, 'columns', [])]]
        didx_all = dd if didx_all is None else pd.concat([didx_all, dd], ignore_index=True)

    # Always write the didx file if we accumulated anything.
    if didx_all is not None:
        th.save(didx_all, os.path.join(didx_loc, f'didx_{didx_fn}.p'))

    if align_didx:
        # Optional alignment step (not used here)
        pass
