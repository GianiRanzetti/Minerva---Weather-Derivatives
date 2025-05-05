import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kurtosis, linregress
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.utils import resample


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
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self(X), y)
            loss.backward()
            optimizer.step()
            if verbose and epoch % (epochs//5) == 0:
                print(f"Epoch {epoch}, loss={loss.item():.4f}")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)

# -----------------------------------------------
# Sensitivity-Based Pruning (SBP) for lag selection
# -----------------------------------------------
def sbp_lag_selection(X, y, hidden_dim, alpha=0.1, B=50, lr=1e-2, epochs=500):
    current = list(range(X.shape[1]))
    improved = True
    while improved and len(current) > 1:
        X_sub = X[:, current]
        # fit full model
        wn_full = WaveletNetwork(len(current), hidden_dim)
        wn_full.fit(X_sub, y, lr=lr, epochs=epochs)
        L_full = nn.MSELoss()(wn_full.predict(X_sub), y).item()
        pvals = []
        for j in range(len(current)):
            # observed SBP
            X_rep = X_sub.clone()
            X_rep[:, j] = X_rep[:, j].mean()
            wn_rep = WaveletNetwork(len(current), hidden_dim)
            wn_rep.fit(X_rep, y, lr=lr, epochs=epochs)
            L_rep = nn.MSELoss()(wn_rep.predict(X_rep), y).item()
            sbp_obs = L_rep - L_full
            # bootstrap distribution under null
            sbp_bs = []
            n = X_sub.shape[0]
            for _ in range(B):
                idx = resample(np.arange(n))
                Xb = X_sub[idx]
                yb = y[idx]
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
        max_p = max(pvals)
        if max_p > alpha:
            drop = pvals.index(max_p)
            current.pop(drop)
        else:
            improved = False
    return current

# -----------------------------------------------
# Hidden dimension selection via K-Fold CV
# -----------------------------------------------
def select_hidden_dim_cv(X, y, lags, candidate_hids, k=5, lr=1e-2, epochs=500):
    best_h, best_mse = None, np.inf
    X_sub = X[:, lags]
    n = X_sub.shape[0]
    for h in candidate_hids:
        mses = []
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        for train_idx, val_idx in kf.split(np.arange(n)):
            Xt, yt = X_sub[train_idx], y[train_idx]
            Xv, yv = X_sub[val_idx], y[val_idx]
            wn = WaveletNetwork(len(lags), h)
            wn.fit(Xt, yt, lr=lr, epochs=epochs)
            mses.append(nn.MSELoss()(wn.predict(Xv), yv).item())
        avg_mse = np.mean(mses)
        if avg_mse < best_mse:
            best_mse, best_h = avg_mse, h
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
# Full WN pipeline
# -----------------------------------------------
def fit_wavelet_model(series, max_lags, candidate_hids, sbp_hid,
                      lr=1e-2, epochs=1000):
    # prepare lag matrix
    vals = series.values
    X = torch.stack([
        torch.tensor(series.shift(i+1).fillna(method='bfill').values,
                     dtype=torch.float32)
        for i in range(max_lags)
    ], dim=1)
    y = torch.tensor(vals, dtype=torch.float32)
    # lag selection
    lags = sbp_lag_selection(X, y, sbp_hid, lr=lr, epochs=epochs)
    # topology selection
    best_h = select_hidden_dim_cv(X, y, lags, candidate_hids, lr=lr, epochs=epochs)
    # final fit
    Xf = X[:, lags]
    wn = WaveletNetwork(len(lags), best_h)
    wn.fit(Xf, y, lr=lr, epochs=epochs)
    k_ser, coefs = estimate_speed_of_mean_reversion(Xf, y, best_h, lr=lr, epochs=epochs)
    return lags, best_h, k_ser, coefs, wn