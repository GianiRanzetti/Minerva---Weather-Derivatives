import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
from scipy.stats import t, norm
import concurrent.futures
from functools import partial


# Add function to determine the best available device
def get_device():
    """Get the best available device: MPS (Mac), CUDA, or CPU"""
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("mps")
    else:
        return torch.device("cuda")

# Get the device once at module level
DEVICE = get_device()
print(f"Using device: {DEVICE}")


class WaveletLayer(nn.Module):
    """Wavelet activation layer for neural networks"""
    def __init__(self, input_dim, output_dim, wavelet='mexh'):
        super(WaveletLayer, self).__init__()
        self.scales = nn.Parameter(torch.randn(output_dim, input_dim))
        self.translations = nn.Parameter(torch.randn(output_dim, input_dim))
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))
        self.wavelet_type = wavelet
        
    def forward(self, x):
        # Apply wavelet activation
        expanded_x = x.unsqueeze(1)  # Shape: [batch, 1, input_dim]
        scales = torch.abs(self.scales) + 0.1  # Ensure positive scales
        
        # Calculate wavelet activation: psi((x-t)/s)
        scaled_diff = (expanded_x - self.translations.unsqueeze(0)) / scales.unsqueeze(0)
        
        # Apply the wavelet function
        if self.wavelet_type == 'mexh':  # Mexican hat wavelet
            # Formula: (1 - x^2) * exp(-x^2/2)
            wavelet_output = (1 - scaled_diff**2) * torch.exp(-scaled_diff**2/2)
        elif self.wavelet_type == 'morlet':  # Morlet wavelet
            # Formula: exp(-x^2/2) * cos(5x)
            wavelet_output = torch.exp(-scaled_diff**2/2) * torch.cos(5*scaled_diff)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")
        
        # Apply weights and sum
        output = torch.sum(self.weights.unsqueeze(0) * wavelet_output, dim=2)
        return output


class WaveletNetwork(nn.Module):
    """Wavelet Network model"""
    def __init__(self, input_dim, hidden_dim, output_dim=1, wavelet='mexh'):
        super(WaveletNetwork, self).__init__()
        self.input_dim = input_dim
        self.wavelet_layer = WaveletLayer(input_dim, hidden_dim, wavelet)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.wavelet_layer(x)
        x = self.output_layer(x)
        return x
    
    def calculate_k(self, x):
        # Clone x and enable gradients
        x_tensor = x.clone().requires_grad_(True)
        # Compute the network output (assumed shape [batch, 1])
        y_pred = self.forward(x_tensor)
        # Create a grad_outputs tensor of ones (same shape as y_pred)
        grad_outputs = torch.ones_like(y_pred)
        # Compute gradients of y_pred with respect to x_tensor (batch-wise)
        grads = torch.autograd.grad(
            outputs=y_pred, 
            inputs=x_tensor, 
            grad_outputs=grad_outputs, 
            retain_graph=False, 
            create_graph=False
        )[0]
        # Assume that T(t) is the first feature (index 0); then:
        k_values = grads[:, 0] - 1
        return k_values.cpu().numpy()


def build_lagged_dataset(data, max_lag=10):
    """
    Build a dataset with lagged values for supervised learning
    
    Parameters:
    -----------
    data : array-like
        Deseasonalized temperature data
    max_lag : int
        Maximum number of lags to include
        
    Returns:
    --------
    X : numpy array
        Input features (lagged values)
    y : numpy array
        Target values (future temperature)
    """
    data = np.array(data)
    n = len(data)
    X = np.zeros((n - max_lag, max_lag))
    y = np.zeros(n - max_lag)
    
    for i in range(max_lag, n):
        X[i - max_lag] = data[i-max_lag:i][::-1]  # Reverse to have most recent lags first
        y[i - max_lag] = data[i]
    
    return X, y


def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32, lr=0.001):
    """
    Train the given model on the dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    X_train, y_train : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    epochs, batch_size, lr : training parameters
        
    Returns:
    --------
    history : dict
        Training history
    """
    import time
    start_time = time.time()
    
    print(f"Starting training with {epochs} epochs, batch size {batch_size}, learning rate {lr}")
    print(f"Model input shape: {X_train.shape}, output shape: {y_train.shape}")
    
    # Move model to device
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert data to torch tensors and move to device
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(DEVICE)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Track training history
    history = {'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    no_improve_count = 0
    print_freq = max(1, min(50, epochs // 20))  # Print at least 20 times during training
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        # Record losses
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Print progress
        if (epoch + 1) % print_freq == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            progress = (epoch + 1) / epochs * 100
            est_total = elapsed / (epoch + 1) * epochs
            est_remaining = est_total - elapsed
            
            print(f"Epoch {epoch+1}/{epochs} ({progress:.1f}%) - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Best Val: {best_val_loss:.6f}, No Improve: {no_improve_count}")
            print(f"Time: {elapsed:.1f}s elapsed, ~{est_remaining:.1f}s remaining")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s - Final Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    return history

def evaluate_model(model, X_test, y_test, dates=None):
    """
    Evaluate the final model on the test set.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model.
    X_test : numpy array
        Test inputs.
    y_test : numpy array
        True test targets.
    dates : array-like, optional
        Dates corresponding to the test set (if available).
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics (MSE, R², mean of k-values, std of k-values).
    predictions : numpy array
        Model predictions on X_test.
    k_values : numpy array
        Estimated k(t) values computed by model.calculate_k.
    residuals : numpy array
        Residuals (y_test - predictions).
    """
    model.eval()
    # Convert test data to tensors.
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(DEVICE)
    
    with torch.no_grad():
        # Get predictions from the model.
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
        
    # Compute evaluation metrics.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate k-values via the model's calculate_k method.
    # Here, we assume that the first feature in X_test corresponds to T(t)
    k_values = model.calculate_k(X_test_tensor)
    
    # Compute residuals (difference between true and predicted values)
    residuals = y_test - y_pred
    
    # Prepare metrics dictionary.
    metrics = {
        'mse': mse,
        'r2': r2,
        'mean_k': np.mean(k_values),
        'std_k': np.std(k_values)
    }
    
    return metrics, y_pred, k_values, residuals

# =============================================================================
# Topology Selection using Minimum Prediction Risk (MPR)
# =============================================================================

def select_optimal_topology(X_train, y_train, X_val, y_val, input_dim, wavelet='mexh',
                            max_hidden=20, epochs=1000, lr=0.001, batch_size=32,
                            risk_increase_factor=1.05, model_type='wavelet'):
    """
    Sequentially adds one hidden unit at a time (starting from 0 HUs, i.e. a linear model)
    and computes the prediction risk (MSE). When adding a hidden unit increases the MSE
    beyond a threshold, the algorithm stops. Returns the optimal number of hidden units.
    """
    best_hidden = 0
    best_risk = None

    # Function to train and compute risk on validation set.
    def train_and_evaluate(hidden_units):
        if model_type.lower() == 'wavelet':
            model = WaveletNetwork(input_dim, hidden_units, output_dim=1, wavelet=wavelet)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        _ = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
        # Evaluate on validation set (use MSE as risk)
        model.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
        with torch.no_grad():
            y_pred = model(X_val_tensor).cpu().numpy().flatten()
        risk = mean_squared_error(y_val, y_pred)
        return risk, model

    # Begin with hidden_units = 0 (linear model)
    risk, _ = train_and_evaluate(0)
    best_risk = risk
    best_hidden = 0
    print(f"Hidden units: 0, Validation MSE: {risk:.6f}")
    
    for hu in range(1, max_hidden + 1):
        risk, _ = train_and_evaluate(hu)
        print(f"Hidden units: {hu}, Validation MSE: {risk:.6f}")
        if risk < best_risk:
            best_risk = risk
            best_hidden = hu
        # If the risk increases too much relative to the best so far, stop the search.
        if risk > best_risk * risk_increase_factor:
            print(f"Stopping topology search at {hu} hidden units (risk increased).")
            break

    print(f"Optimal hidden units selected: {best_hidden} with validation MSE: {best_risk:.6f}")
    return best_hidden

# =============================================================================
# Lag (Variable) Selection using Bootstrapped Sensitivity-Based Pruning (SBP)
# =============================================================================

def compute_SBP(model, X_data, y_data, variable_index):
    """
    Compute the sensitivity-based pruning (SBP) measure for the specified variable index.
    SBP(x_j) = L_n(X, w_hat) - L_n(tilde{X}^{(j)}, w_hat)
    where tilde{X}^{(j)} is X with the j-th column replaced by its mean.
    The loss L_n is computed as the mean squared error over X_data.
    """
    # Compute loss on original data using the current model.
    model.eval()
    X_tensor = torch.FloatTensor(X_data).to(DEVICE)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    loss_original = mean_squared_error(y_data, y_pred)

    # Replace column j with its mean.
    X_modified = np.array(X_data, copy=True)
    col_mean = np.mean(X_modified[:, variable_index])
    X_modified[:, variable_index] = col_mean

    X_mod_tensor = torch.FloatTensor(X_modified).to(DEVICE)
    with torch.no_grad():
        y_pred_mod = model(X_mod_tensor).cpu().numpy().flatten()
    loss_modified = mean_squared_error(y_data, y_pred_mod)

    return loss_original - loss_modified

def bootstrap_SBP_worker(b, X, y, active_lags, model_constructor, epochs, lr, batch_size):
    n = X.shape[0]
    bootstrap_indices = np.random.choice(n, size=n, replace=True)
    X_boot = X[bootstrap_indices, :]
    y_boot = y[bootstrap_indices]
    current_input_dim = len(active_lags)
    model = model_constructor(current_input_dim).to(DEVICE)
    _ = train_model(model, X_boot, y_boot, X_boot, y_boot, epochs=epochs, lr=lr, batch_size=batch_size)
    sbp_values = []
    # Here, iterate over the columns of the bootstrapped dataset. They are 0-indexed.
    for i in range(X_boot.shape[1]):
        sbp_value = compute_SBP(model, X_boot, y_boot, i)
        sbp_values.append(sbp_value)
    # Optionally, we can map these back to the original lag indices
    return {active_lags[i]: sbp_values[i] for i in range(len(active_lags))}

def bootstrap_SBP(X, y, active_lags, model_constructor, B=200, epochs=500, lr=0.001, batch_size=32):
    SBP_results = {j: [] for j in active_lags}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(bootstrap_SBP_worker, b, X, y, active_lags, model_constructor, epochs, lr, batch_size) for b in range(B)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for j in active_lags:
                SBP_results[j].append(result[j])
    return SBP_results

def compute_p_value(sbp_values, B):
    """
    Compute a two-sided p-value from the bootstrap distribution of SBP values.
    Here we use a t-statistic: t = mean / (std / sqrt(B)) and then p = 2 * (1 - CDF(|t|)).
    """
    mean_val = np.mean(sbp_values)
    std_val = np.std(sbp_values, ddof=1)
    if std_val == 0:
        return 1.0  # if no variability, no significance
    t_stat = mean_val / (std_val / np.sqrt(B))
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=B-1))
    return p_val

# Define a helper model constructor that returns an untrained model.
def wavelet_model_constructor(input_dim, wavelet='mexh'):
    """Return an untrained WaveletNetwork model with the specified input dimension and wavelet type."""
    return WaveletNetwork(input_dim, hidden_dim=10, output_dim=1, wavelet=wavelet)

def select_significant_lags(X, y, wavelet='mexh', model_type='wavelet',
                            initial_active_lags=None, B=200, epochs=500, lr=0.001,
                            batch_size=32, pval_threshold=0.1, risk_threshold=1.05):
    """
    Implements the variable selection algorithm from the book.
    """
    if initial_active_lags is None:
        active_lags = list(range(X.shape[1]))
    else:
        active_lags = initial_active_lags[:]
    
    # Use the global function with partial to fix the wavelet parameter.
    if model_type.lower() == 'wavelet':
        model_constructor = partial(wavelet_model_constructor, wavelet=wavelet)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Get an initial risk on the full training dataset.
    X_current = X[:, active_lags]
    X_train, X_val, y_train, y_val = train_test_split(X_current, y, test_size=0.2, shuffle=False)

    model_full = model_constructor(len(active_lags)).to(DEVICE)
    _ = train_model(model_full, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
    model_full.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    with torch.no_grad():
        y_pred_full = model_full(X_val_tensor).cpu().numpy().flatten()
    best_risk = mean_squared_error(y_val, y_pred_full)
    print(f"Initial risk with all lags {active_lags}: {best_risk:.6f}")
    
    improved = True
    while improved and len(active_lags) > 1:
        improved = False
        # Compute bootstrap SBP for the current set of active lags.
        SBP_dict = bootstrap_SBP(X[:, active_lags], y, active_lags, model_constructor,
                                B=B, epochs=epochs, lr=lr, batch_size=batch_size)
        p_values = {}
        for j in active_lags:
            p_values[j] = compute_p_value(SBP_dict[j], B)
            print(f"Lag {j}: mean SBP = {np.mean(SBP_dict[j]):.6f}, p-value = {p_values[j]:.4f}")
        
        # Identify candidate lag to remove: the one with highest p-value (if above threshold)
        candidates = {j: p for j, p in p_values.items() if p > pval_threshold}
        if not candidates:
            print("All remaining lags are statistically significant.")
            break
        lag_to_remove = max(candidates, key=candidates.get)
        print(f"Candidate lag to remove: {lag_to_remove} with p-value {candidates[lag_to_remove]:.4f}")
        
        # Test removal: build new active set without this lag.
        new_active_lags = active_lags[:]
        new_active_lags.remove(lag_to_remove)
        X_new = X[:, new_active_lags]
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y, test_size=0.2, shuffle=False)
        model_new = model_constructor(len(new_active_lags)).to(DEVICE)
        _ = train_model(model_new, X_train_new, y_train_new, X_val_new, y_val_new,
                        epochs=epochs, lr=lr, batch_size=batch_size)
        model_new.eval()
        X_val_new_tensor = torch.FloatTensor(X_val_new).to(DEVICE)
        with torch.no_grad():
            y_pred_new = model_new(X_val_new_tensor).cpu().numpy().flatten()
        new_risk = mean_squared_error(y_val_new, y_pred_new)
        print(f"Risk after removing lag {lag_to_remove}: {new_risk:.6f} (previous: {best_risk:.6f})")
        
        # If removal does not increase the risk too much, accept removal.
        if new_risk <= best_risk * risk_threshold:
            print(f"Removing lag {lag_to_remove} accepted.")
            active_lags = new_active_lags
            best_risk = new_risk
            improved = True
        else:
            print(f"Removal of lag {lag_to_remove} rejected (risk increased too much).")
            break

    print(f"Selected lags after variable selection: {active_lags}")
    X_pruned = X[:, active_lags]
    return X_pruned, active_lags

def estimate_mean_reversion(deseasonalized_temp, dates=None, max_lag=10, test_size=0.2, 
                            hidden_dim_range=[0, 5, 10, 15, 20], wavelet='mexh', 
                            model_type='wavelet', do_variable_selection=True,
                            B=200, epochs=1000, lr=0.001, batch_size=32):
    """
    Extended function to estimate mean reversion using a wavelet network.
    It integrates:
       1) Building a lagged dataset,
       2) (Optionally) pruning irrelevant lags using bootstrapped sensitivity-based pruning,
       3) Selecting the optimal topology (number of hidden units) via the MPR principle,
       4) Training the final model and evaluating it.
       
    Returns a dictionary with the final model, metrics, predictions, k-values, and history.
    """
    # Step 1: Build dataset with lagged values.
    X, y = build_lagged_dataset(deseasonalized_temp, max_lag)
    
    # Step 2: (Optional) Relevant-lag (variable) selection.
    if do_variable_selection:
        print("Starting variable selection using bootstrapped SBP...")
        X_pruned, active_lags = select_significant_lags(X, y, wavelet=wavelet, model_type=model_type,
                                                        B=B, epochs=epochs//2, lr=lr, batch_size=batch_size)
    else:
        X_pruned = X
        active_lags = list(range(X.shape[1]))
    
    # Step 3: Split data into train, validation, and test sets.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_pruned, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")
    print(f"Active lags after variable selection: {active_lags}")
    print(f"Input shape after variable selection: {X_pruned.shape}")

    # Step 4: Topology selection using MPR on the training/validation split.
    input_dim = X_pruned.shape[1]
    optimal_hidden_units = select_optimal_topology(X_train, y_train, X_val, y_val, input_dim,
                                                   wavelet=wavelet, max_hidden=max(hidden_dim_range),
                                                   epochs=epochs, lr=lr, batch_size=batch_size,
                                                   model_type=model_type)
    
    # Step 5: Create final model with the selected topology and train on full training set.
    if model_type.lower() == 'wavelet':
        model = WaveletNetwork(input_dim, optimal_hidden_units, output_dim=1, wavelet=wavelet)
        model_name = f"Wavelet Network ({wavelet}), hidden_dim={optimal_hidden_units}"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Training final model: {model_name}")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
    
    # Step 6: Evaluate final model.
    print("Evaluating final model on test set...")
    metrics, predictions, k_values, residuals = evaluate_model(model, X_test, y_test, dates)
    
    print(f"\nFinal Model: {model_name}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Mean k(t): {metrics['mean_k']:.6f}")
    print(f"Std k(t): {metrics['std_k']:.6f}")
    
    results = {
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions,
        'k_values': k_values,
        'active_lags': active_lags,
        'history': history,
        'residuals': residuals
    }
    
    return results