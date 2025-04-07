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
    """Multi-Layer Perceptron model"""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert data to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Track training history
    history = {'train_loss': [], 'val_loss': []}
    
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
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
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
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
    
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


def estimate_mean_reversion(deseasonalized_temp, dates=None, max_lag=10, test_size=0.2, 
                          hidden_dim=20, wavelet='mexh', model_type='wavelet'):
    """
    Main function to estimate mean reversion speed from deseasonalized temperature data
    
    Parameters:
    -----------
    deseasonalized_temp : array-like
        Deseasonalized temperature data
    dates : array-like, optional
        Dates corresponding to temperature data
    max_lag : int
        Maximum number of lags to include initially
    test_size : float
        Proportion of data to use for testing
    hidden_dim : int
        Number of hidden neurons/wavelets
    wavelet : str
        Wavelet type ('mexh' or 'morlet')
    model_type : str
        Model type ('wavelet' or 'mlp')
        
    Returns:
    --------
    results : dict
        Dictionary containing trained model and evaluation results
    """
    # Build dataset with lagged values
    X, y = build_lagged_dataset(deseasonalized_temp, max_lag)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
    
    # Create model based on specified type
    if model_type.lower() == 'wavelet':
        model = WaveletNetwork(X_train.shape[1], hidden_dim, output_dim=1, wavelet=wavelet)
        model_name = f"Wavelet Network ({wavelet})"
    else:
        model = MLP(X_train.shape[1], hidden_dim, output_dim=1)
        model_name = "Multi-Layer Perceptron"
    
    # Train model
    print(f"Training {model_name}...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=1000)
    
    # Feature selection using sensitivity-based pruning
    X_tensor = torch.FloatTensor(X_train)
    significant_lags, sensitivities = sensitivity_based_pruning(model, X_tensor)
    
    print(f"Significant lags: {significant_lags}")
    
    # Plot feature importance based on sensitivities
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sensitivities)), sensitivities)
    plt.xlabel('Lag')
    plt.ylabel('Normalized Sensitivity')
    plt.title('Feature Importance by Lag')
    plt.xticks(range(len(sensitivities)), [f'T(t-{i})' for i in range(len(sensitivities))])
    plt.grid(True)
    plt.show()
    
    # If we have dates, prepare them for plotting
    test_dates = None
    if dates is not None:
        # Make sure dates align with test data
        test_dates = dates[max_lag:][-(len(X_test)):]
    
    # Evaluate the model
    print(f"Evaluating {model_name}...")
    metrics, predictions, k_values, residuals = evaluate_model(model, X_test, y_test, test_dates)
    
    # Print metrics
    print(f"\nModel: {model_name}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Mean k(t): {metrics['mean_k']:.6f}")
    print(f"Std k(t): {metrics['std_k']:.6f}")
    
    # Return results
    results = {
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions,
        'k_values': k_values,
        'significant_lags': significant_lags,
        'history': history,
        'residuals': residuals
    }
    
    return results




