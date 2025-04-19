import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import pandas as pd

torch.manual_seed(0)

class WaveletNetwork(nn.Module):
    """
    A Wavelet Neural Network with one hidden layer of Mexican-hat wavelets,
    plus a direct linear connection from inputs to output.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # direct linear connection weights (no bias)
        self.linear_x = nn.Linear(input_dim, 1, bias=False)
        # wavelet-to-output weights and output bias
        self.w2 = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        # translation (b) and scale (a) parameters for each wavelet and each input
        self.b = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.log_a = nn.Parameter(torch.zeros(hidden_dim, input_dim))  # will exponentiate to ensure positivity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct = self.linear_x(x).squeeze(-1)
        a = torch.exp(self.log_a)
        x_exp = x.unsqueeze(1)               # (batch, 1, input_dim)
        b_exp = self.b.unsqueeze(0)          # (1, hidden_dim, input_dim)
        a_exp = a.unsqueeze(0)               # (1, hidden_dim, input_dim)
        z = (x_exp - b_exp) / a_exp           # (batch, hidden_dim, input_dim)
        phi = (1 - z**2) * torch.exp(-0.5 * z**2)
        C = torch.prod(phi, dim=2)           # (batch, hidden_dim)
        wavelet_out = C @ self.w2            # (batch,)
        return direct + wavelet_out + self.bias

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float=1e-2,
            epochs: int=1000, verbose: bool=False) -> None:
        """
        Train the network on (X, y) using Adam optimizer and MSE loss.
        """
        self.train()
        opt = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for ep in range(epochs):
            opt.zero_grad()
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            if verbose and ep % (epochs//5) == 0:
                print(f"Epoch {ep}, loss={loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def partial_coefs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂y_pred/∂x_j for each sample and input dimension.
        Returns (n_samples, input_dim).
        """
        self.eval()
        X_req = X.clone().detach().requires_grad_(True)
        y_pred = self.forward(X_req)
        grads = torch.zeros_like(X_req)
        for i in range(len(X_req)):
            grads_i = torch.autograd.grad(
                y_pred[i], X_req, retain_graph=True, only_inputs=True
            )[0][i]
            grads[i] = grads_i
        return grads

# Utility functions

def train_val_split(X, y, val_fraction=0.2):
    n = len(X)
    split = int(n * (1 - val_fraction))
    return X[:split], y[:split], X[split:], y[split:]


def select_hidden_dim(
    X: torch.Tensor, y: torch.Tensor, lags: list,
    candidate_dims: list, lr: float, epochs: int
) -> int:
    """
    Choose best hidden_dim by train/validation split minimizing val MSE.
    """
    X_sub = X[:, lags]
    X_tr, y_tr, X_va, y_va = train_val_split(X_sub, y)
    best_dim, best_loss = None, float('inf')
    for hd in candidate_dims:
        wn = WaveletNetwork(input_dim=len(lags), hidden_dim=hd)
        wn.fit(X_tr, y_tr, lr=lr, epochs=epochs)
        val_loss = nn.MSELoss()(wn.predict(X_va), y_va).item()
        if val_loss < best_loss:
            best_loss, best_dim = val_loss, hd
    return best_dim


def select_lags_backward_elimination(
    X: torch.Tensor, y: torch.Tensor, max_lags: int,
    hidden_dim: int, lr: float, epochs: int, val_fraction=0.2
) -> list:
    """
    Iteratively remove the least important lag (lowest |coef|) if(val MSE
    does not increase). Returns list of selected lag indices.
    """
    current = list(range(max_lags))
    X_sub = X[:, current]
    X_tr, y_tr, X_va, y_va = train_val_split(X_sub, y, val_fraction)
    # initial loss
    wn = WaveletNetwork(input_dim=len(current), hidden_dim=hidden_dim)
    wn.fit(X_tr, y_tr, lr=lr, epochs=epochs)
    best_loss = nn.MSELoss()(wn.predict(X_va), y_va).item()
    improved = True
    while improved and len(current) > 1:
        # train full on current
        wn_full = WaveletNetwork(input_dim=len(current), hidden_dim=hidden_dim)
        wn_full.fit(X_tr, y_tr, lr=lr, epochs=epochs)
        coefs = wn_full.partial_coefs(X_tr).abs().mean(dim=0)
        # identify lag to drop
        drop_idx = torch.argmin(coefs).item()
        cand = current[:drop_idx] + current[drop_idx+1:]
        X_sub_c = X[:, cand]
        X_tr_c, y_tr_c, X_va_c, y_va_c = train_val_split(X_sub_c, y, val_fraction)
        wn_cand = WaveletNetwork(input_dim=len(cand), hidden_dim=hidden_dim)
        wn_cand.fit(X_tr_c, y_tr_c, lr=lr, epochs=epochs)
        loss_cand = nn.MSELoss()(wn_cand.predict(X_va_c), y_va_c).item()
        if loss_cand <= best_loss:
            best_loss = loss_cand
            current = cand
            X_tr, y_tr, X_va, y_va = X_tr_c, y_tr_c, X_va_c, y_va_c
            improved = True
        else:
            improved = False
    return current


def estimate_speed_of_mean_reversion(
    X: torch.Tensor, y: torch.Tensor, hidden_dim: int,
    lr: float=1e-2, epochs: int=1000
) -> (torch.Tensor, torch.Tensor):
    """
    Trains a WN on X->y, then returns k(t)=a1(t)-1 and all a_i(t).
    """
    wn = WaveletNetwork(input_dim=X.size(1), hidden_dim=hidden_dim)
    wn.fit(X, y, lr=lr, epochs=epochs)
    coefs = wn.partial_coefs(X)
    k = coefs[:,0] - 1
    return k, coefs


def fit_wavelet_model(
    series: pd.Series, max_lags: int,
    candidate_hids: list, lr: float=1e-2, epochs: int=1000, return_model=False
):
    """
    Full pipeline: detrend & deseasonalize outside,
    then:
      1. Select lags via backward elimination
      2. Select hidden_dim via val-split
      3. Estimate k(t)
    Returns selected_lags, best_hidden, k_series, full_coefs
    """
    # Build lag matrix X and target y
    X_list = []
    for i in range(max_lags):
        X_list.append(torch.tensor(series.shift(i+1).iloc[max_lags:].values, dtype=torch.float32))
    X = torch.stack(X_list, dim=1)
    y = torch.tensor(series.iloc[max_lags:].values, dtype=torch.float32)
    # 1. lag selection
    lags = select_lags_backward_elimination(X, y, max_lags, candidate_hids[0], lr, epochs)
    # 2. topology selection
    best_h = select_hidden_dim(X, y, lags, candidate_hids, lr, epochs)
    # build & train the final model
    wn = WaveletNetwork(input_dim=len(lags), hidden_dim=best_h)
    X_final = X[:, lags]     # X is the lagged‐feature matrix you built earlier
    wn.fit(X_final, y, lr=lr, epochs=epochs)
    # 3. k estimation
    k_series, coefs = estimate_speed_of_mean_reversion(X[:, lags], y, best_h, lr, epochs)
    if return_model:
        return lags, best_h, k_series, coefs, wn, max_lags
    else:
        return lags, best_h, k_series, coefs
