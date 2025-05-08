#!/usr/bin/env python3
"""
Wavelet-based temperature mean reversion model for a weather station near Los Angeles.
Verbose logging added to SBP, CV, and full pipeline.
"""
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim

# Select device for computation (force CPU)
device = torch.device("cpu")
print("DEBUG: Forcing CPU for computation.")

from scipy.stats import kurtosis, linregress
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.utils import resample

# reproducibility
torch.manual_seed(0)
np.random.seed(0)

class WaveletNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.w2 = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.log_a = nn.Parameter(torch.zeros(hidden_dim, input_dim))

    def forward(self, x):
        direct = self.linear(x).squeeze(-1)
        a = torch.exp(self.log_a)
        z = (x.unsqueeze(1) - self.b.unsqueeze(0)) / a.unsqueeze(0)
        phi = (1 - z**2) * torch.exp(-0.5 * z**2)
        C = torch.prod(phi, dim=2)
        wave = C @ self.w2
        return direct + wave + self.bias

    def fit(self, X, y, lr=1e-2, epochs=1000, verbose=False):
        # Move model and data to device
        self.to(device)
        X = X.to(device)
        y = y.to(device)

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            if verbose and epoch % max(1, epochs//5) == 0:
                print(f"  [WN] Epoch {epoch}/{epochs}, loss={loss.item():.6f}")

        return                  


    def predict(self, X):
        # Move input to the same device as the model
        X = X.to(device)

        self.eval()
        with torch.no_grad():
            return self(X)

# -----------------------------------------------
# Sensitivity-Based Pruning (SBP) for lag selection
# -----------------------------------------------
def sbp_lag_selection(X, y, hidden_dim, alpha=0.1, B=20, lr=1e-2, epochs=300, verbose=False):
    current = list(range(X.shape[1]))
    if verbose: print(f"[SBP] Starting with lags: {current}")
    improved = True
    iter_count = 0
    while improved and len(current) > 1:
        iter_count += 1
        if verbose: print(f"[SBP] Iter {iter_count}, lags: {current}")
        X_sub = X[:, current]
        wn_full = WaveletNetwork(len(current), hidden_dim)
        wn_full.fit(X_sub, y, lr=lr, epochs=epochs, verbose=verbose)
        L_full = nn.MSELoss()(wn_full.predict(X_sub), y).item()
        pvals = []
        for j in range(len(current)):
            X_rep = X_sub.clone()
            X_rep[:, j] = X_rep[:, j].mean()
            wn_rep = WaveletNetwork(len(current), hidden_dim)
            wn_rep.fit(X_rep, y, lr=lr, epochs=epochs)
            L_rep = nn.MSELoss()(wn_rep.predict(X_rep), y).item()
            sbp_obs = L_rep - L_full
            sbp_bs = []
            n = X_sub.shape[0]
            for _ in range(B):
                idx = resample(np.arange(n))
                Xb = X_sub[idx]; yb = y[idx]
                wn_b = WaveletNetwork(len(current), hidden_dim)
                wn_b.fit(Xb, yb, lr=lr, epochs=epochs)
                Lb_full = nn.MSELoss()(wn_b.predict(Xb), yb).item()
                Xb_rep = Xb.clone()
                Xb_rep[:, j] = X_sub[:, j].mean()
                wn_brep = WaveletNetwork(len(current), hidden_dim)
                wn_brep.fit(Xb_rep, yb, lr=lr, epochs=epochs)
                Lb_rep = nn.MSELoss()(wn_brep.predict(Xb_rep), yb).item()
                sbp_bs.append(Lb_rep - Lb_full)
            pval = (np.sum(np.array(sbp_bs) >= sbp_obs) + 1) / (B + 1)
            pvals.append(pval)
            if verbose: print(f"   [SBP] Lag {current[j]} p={pval:.3f}")
        max_p = max(pvals)
        if max_p > alpha:
            drop = pvals.index(max_p)
            if verbose: print(f"[SBP] Dropping lag {current[drop]} (p={max_p:.3f} > {alpha})")
            current.pop(drop)
        else:
            if verbose: print("[SBP] No lag to drop; finished.")
            improved = False
    if verbose: print(f"[SBP] Final lags: {current}")
    return current

# -----------------------------------------------
# Hidden dimension selection via K-Fold CV
# -----------------------------------------------
def select_hidden_dim_cv(X, y, lags, candidate_hids, k=5, lr=1e-2, epochs=300, verbose=False):
    best_h, best_mse = None, np.inf
    X_sub = X[:, lags]; n = X_sub.shape[0]
    for h in candidate_hids:
        if verbose: print(f"[CV] Testing h={h}")
        mses = []
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        for fold, (ti, vi) in enumerate(kf.split(np.arange(n)),1):
            Xt, yt = X_sub[ti], y[ti]
            Xv, yv = X_sub[vi], y[vi]
            wn = WaveletNetwork(len(lags), h)
            wn.fit(Xt, yt, lr=lr, epochs=epochs)
            mse = nn.MSELoss()(wn.predict(Xv), yv).item()
            mses.append(mse)
            if verbose: print(f"  [CV] fold {fold}, mse={mse:.6f}")
        avg = np.mean(mses)
        if verbose: print(f"[CV] h={h}, avg mse={avg:.6f}")
        if avg < best_mse: best_mse, best_h = avg, h
    if verbose: print(f"[CV] Selected h={best_h} (mse={best_mse:.6f})")
    return best_h

# -----------------------------------------------
# Estimate speed of mean reversion k(t)
# -----------------------------------------------
def estimate_speed_of_mean_reversion(X, y, hidden_dim, lr=1e-2, epochs=1000):
    wn = WaveletNetwork(X.shape[1], hidden_dim)
    wn.fit(X, y, lr=lr, epochs=epochs)
    X_req = X.clone().detach().requires_grad_(True)
    y_pred = wn(X_req)
    grads = torch.autograd.grad(y_pred.sum(), X_req)[0]
    k = grads[:, 0] - 1
    return k, grads

# -----------------------------------------------
# Full WN pipeline with verbosity
# -----------------------------------------------
def fit_wavelet_model(series, max_lags, candidate_hids, sbp_hid,
                      lr=1e-2, epochs=1000, verbose=True):
    if verbose: print("=== Wavelet Mean-Reversion Pipeline ===")
    vals = series.values
    X = torch.stack([
        torch.tensor(series.shift(i+1).fillna(method='bfill').values,
                     dtype=torch.float32)
        for i in range(max_lags)
    ], dim=1)
    y = torch.tensor(vals, dtype=torch.float32)

    # Move data to device
    X = X.to(device)
    y = y.to(device)

    if verbose: print("--> SBP lag selection")
    lags = sbp_lag_selection(X, y, sbp_hid, lr=lr, epochs=epochs, verbose=verbose)
    if verbose: print("--> Hidden-dim selection via CV")
    best_h = select_hidden_dim_cv(X, y, lags, candidate_hids, lr=lr, epochs=epochs, verbose=verbose)
    if verbose: print(f"--> Final model: lags={lags}, h={best_h}")
    Xf = X[:, lags]
    wn = WaveletNetwork(len(lags), best_h)
    wn.fit(Xf, y, lr=lr, epochs=epochs, verbose=verbose)
    k_ser, coefs = estimate_speed_of_mean_reversion(Xf, y, best_h, lr=lr, epochs=epochs)
    if verbose: print("=== Pipeline complete ===")
    return lags, best_h, k_ser, coefs, wn
