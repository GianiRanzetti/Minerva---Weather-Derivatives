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
from scipy.stats import norm
import scipy.stats as stats


# Add function to determine the best available device
def get_device():
    """Get the best available device: MPS (Mac), CUDA, or CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

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
        """Calculate the mean reversion parameter k(t)"""
        x_tensor = x.clone().requires_grad_(True)
        y_pred = self.forward(x_tensor)
        
        # Calculate gradients with respect to the first lag (T_t)
        grads = []
        for i in range(len(x_tensor)):
            # Zero all previous gradients
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            
            # Backward for this sample
            y_pred[i].backward(retain_graph=True)
            
            # Get gradient for T(t) (assuming it's the first feature)
            grad_t = x_tensor.grad[i, 0].item()
            grads.append(grad_t)
        
        # k(t) = ∂T(t+1)/∂T(t) - 1
        k_values = np.array(grads) - 1
        return k_values


class MLP(nn.Module):
    """Standard MLP model for comparison"""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def calculate_k(self, x):
        """Calculate the mean reversion parameter k(t)"""
        x_tensor = x.clone().requires_grad_(True)
        y_pred = self.forward(x_tensor)
        
        # Calculate gradients with respect to the first lag (T_t)
        grads = []
        for i in range(len(x_tensor)):
            # Zero all previous gradients
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            
            # Backward for this sample
            y_pred[i].backward(retain_graph=True)
            
            # Get gradient for T(t) (assuming it's the first feature)
            grad_t = x_tensor.grad[i, 0].item()
            grads.append(grad_t)
        
        # k(t) = ∂T(t+1)/∂T(t) - 1
        k_values = np.array(grads) - 1
        return k_values


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


def sensitivity_based_pruning(model, X_tensor, threshold=0.01):
    """
    Perform sensitivity-based pruning to select significant lags
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X_tensor : torch.Tensor
        Input tensor
    threshold : float
        Sensitivity threshold for keeping variables
        
    Returns:
    --------
    significant_lags : list
        Indices of significant lags
    """
    X_test = X_tensor.clone().requires_grad_(True)
    y_pred = model(X_test)
    
    # Calculate average sensitivity for each lag
    sensitivities = []
    for j in range(X_test.shape[1]):
        X_test.grad = None  # Clear gradients
        
        # Get output with respect to specific lag
        y_sum = y_pred.sum()
        y_sum.backward(retain_graph=True)
        
        # Get average sensitivity for this lag
        avg_sensitivity = torch.abs(X_test.grad[:, j]).mean().item()
        sensitivities.append(avg_sensitivity)
    
    # Normalize sensitivities
    normalized_sens = np.array(sensitivities) / np.max(sensitivities)
    
    # Select lags with sensitivity above threshold
    significant_lags = [i for i, sens in enumerate(normalized_sens) if sens > threshold]
    
    return significant_lags, normalized_sens


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
    Evaluate the model and visualize results
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X_test, y_test : numpy arrays
        Test data
    dates : array-like, optional
        Dates corresponding to test data for plotting
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    
    # Calculate k(t) values
    k_values = model.calculate_k(X_test_tensor)
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(14, 15))
    
    # Plot 1: Predicted vs Actual
    axs[0].plot(y_test, label='Actual', alpha=0.7)
    axs[0].plot(y_pred, label='Predicted', alpha=0.7)
    axs[0].set_title(f'Actual vs Predicted Temperature (MSE: {mse:.4f}, R²: {r2:.4f})')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Deseasonalized Temperature')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Mean Reversion Parameter k(t)
    if dates is not None:
        axs[1].plot(dates[-len(k_values):], k_values)
    else:
        axs[1].plot(k_values)
    axs[1].set_title('Estimated Mean Reversion Parameter k(t)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('k(t)')
    axs[1].grid(True)
    
    # Plot 3: Residual Analysis
    plot_acf(residuals, lags=40, ax=axs[2])
    axs[2].set_title('Autocorrelation of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # QQ plot for residuals
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True)
    plt.show()
    
    # Histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, density=True, alpha=0.7)
    
    # Add a normal curve for comparison
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
    
    # Return metrics
    metrics = {
        'mse': mse,
        'r2': r2,
        'mean_k': np.mean(k_values),
        'std_k': np.std(k_values)
    }
    
    return metrics, y_pred, k_values, residuals


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import t
import copy

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
            model = MLP(input_dim, hidden_units, output_dim=1)
        
        # Train the model (using your existing train_model function)
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

def bootstrap_SBP(X, y, active_lags, model_constructor, B=200, epochs=500, lr=0.001, batch_size=32):
    """
    For each active lag (each column index in active_lags), perform B bootstrap replications
    and compute the SBP measure. Returns a dictionary mapping lag index (from active_lags)
    to a list of SBP values.
    """
    SBP_results = {j: [] for j in active_lags}
    n = X.shape[0]
    for b in range(B):
        # Create bootstrap sample indices
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[bootstrap_indices, :]
        y_boot = y[bootstrap_indices]
        # Build model on bootstrap sample (using the current set of active lags)
        current_input_dim = len(active_lags)
        model = model_constructor(current_input_dim).to(DEVICE)
        # Train on the bootstrap sample
        _ = train_model(model, X_boot, y_boot, X_boot, y_boot, epochs=epochs, lr=lr, batch_size=batch_size)
        # For each lag, compute SBP
        for j in active_lags:
            sbp_value = compute_SBP(model, X_boot, y_boot, active_lags.index(j))
            SBP_results[j].append(sbp_value)
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

def select_significant_lags(X, y, wavelet='mexh', model_type='wavelet',
                            initial_active_lags=None, B=200, epochs=500, lr=0.001,
                            batch_size=32, pval_threshold=0.1, risk_threshold=1.05):
    """
    Implements the variable selection algorithm from the book.
    Iteratively, for the current set of active lags, compute the SBP measure (via bootstrapping)
    and its p-value. Remove the lag with the largest p-value above the threshold if doing so does
    not deteriorate the prediction risk (MSE) by more than risk_threshold times.
    
    Returns:
      X_pruned: the dataset with only the selected lags
      active_lags: a list of indices (columns) that remain.
    """
    if initial_active_lags is None:
        active_lags = list(range(X.shape[1]))
    else:
        active_lags = initial_active_lags[:]
    
    # Define a helper model constructor that returns an untrained model.
    def model_constructor(input_dim):
        if model_type.lower() == 'wavelet':
            return WaveletNetwork(input_dim, hidden_dim=10, output_dim=1, wavelet=wavelet)
        else:
            return MLP(input_dim, hidden_dim=10, output_dim=1)
    
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
        model = MLP(input_dim, optimal_hidden_units, output_dim=1)
        model_name = f"MLP with hidden_dim={optimal_hidden_units}"
    
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