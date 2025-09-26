# Minerva Weather Derivatives Project

A comprehensive weather derivatives pricing and forecasting project developed for the Minerva Student Association. This project implements multiple forecasting models to price Heating Degree Day (HDD) and Cooling Degree Day (CDD) weather derivatives, heavily guided by the academic reference "Weather derivatives modeling and pricing weather-related risk" by Antonis K. Alexandridis and Achilleas Zapranis.

## 🌡️ Project Overview

This repository contains a complete implementation of weather derivatives pricing using three different forecasting methodologies:

- **SARIMA (Seasonal ARIMA)** - Traditional time series forecasting
- **LSTM Neural Networks** - Deep learning approach with CNN-LSTM hybrid
- **Levy Process Model** - Advanced stochastic modeling with wavelet analysis

The project focuses on pricing weather derivatives based on temperature forecasts, specifically HDD and CDD contracts, which are fundamental instruments in weather risk management.

## 📁 Repository Structure

```
Minerva---Weather-Derivatives/
├── pricer.ipynb                    # Main pricing notebook comparing all models
├── Pricing.ipynb                   # Basic pricing implementation
├── CNN-LSTM Model/
│   └── Neural-Networks.ipynb       # Deep learning implementation
├── SARIMA/
│   └── SARIMA Final.ipynb          # SARIMA forecasting model
├── Levy_process_model/
│   ├── AR_model/                   # Autoregressive model with wavelets
│   │   ├── main_WA_model_AR.ipynb
│   │   ├── WA_seasonal_mean.py
│   │   └── forecast_AR.ipynb
│   └── requirements.txt            # Dependencies
├── Forecasts/                      # Model outputs
│   ├── SARIMA_Forecast.csv
│   ├── predictions_with_ci.csv
│   └── forecast_mc_results.json
├── Plots/                          # Visualization outputs
│   ├── ACFPACF*.png
│   ├── *forecast*.png
│   └── SARIMAForecastVs*.png
└── Books/                          # Academic reference material
    └── Weather derivatives modeling and pricing weather-related risk.pdf
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages (see `Levy_process_model/requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Minerva---Weather-Derivatives.git
cd Minerva---Weather-Derivatives
```

2. Install dependencies:
```bash
pip install -r Levy_process_model/requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### Running the Analysis

1. **Start with the main pricing notebook**: `pricer.ipynb`
2. **Explore individual models**:
   - `SARIMA/SARIMA Final.ipynb` for time series analysis
   - `CNN-LSTM Model/Neural-Networks.ipynb` for deep learning
   - `Levy_process_model/AR_model/main_WA_model_AR.ipynb` for stochastic modeling

## 📊 Weather Derivatives

### HDD (Heating Degree Days)
- **Definition**: Max(0, Reference Temperature - Daily Temperature)
- **Use Case**: Heating energy demand hedging
- **Payout**: When temperature is below reference temperature

### CDD (Cooling Degree Days)
- **Definition**: Max(0, Daily Temperature - Reference Temperature)
- **Use Case**: Cooling energy demand hedging
- **Payout**: When temperature is above reference temperature

## 🔬 Forecasting Models

### 1. SARIMA Model
- **Type**: Seasonal Autoregressive Integrated Moving Average
- **Features**: Stationarity tests, seasonal decomposition, ACF/PACF analysis
- **Best for**: Traditional time series with clear seasonal patterns

### 2. LSTM Neural Network
- **Type**: Long Short-Term Memory with CNN components
- **Features**: Deep learning, confidence intervals
- **Best for**: Complex non-linear patterns in temperature data

### 3. Levy Process Model (UO Model)
- **Type**: Stochastic process with wavelet decomposition
- **Features**: Mean reversion, seasonal mean estimation, Monte Carlo simulation
- **Best for**: Advanced quantitative modeling with jump processes

## 📈 Model Performance

Based on the analysis results:

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| UO Model | 2.69 | 3.45 | 10.35% | 0.166 |
| SARIMA | 4.10 | 5.27 | 17.44% | -0.947 |
| LSTM | 5.27 | 6.58 | 23.44% | -2.003 |

*The UO Model shows the best overall performance with the highest R² score and lowest error metrics.*

## 💰 Pricing Function

The core pricing function calculates derivative prices based on:

```python
def price_weather_derivative(forecast_series, historical_data, strike, notional, reference_temp=None):
    """
    Prices HDD and CDD weather derivatives based on temperature forecasts.
    
    Parameters:
    - forecast_series: Temperature forecast data
    - historical_data: Historical temperature data
    - strike: Strike price of the derivative
    - notional: Notional value for payout calculation
    - reference_temp: Reference temperature (auto-calculated if None)
    
    Returns:
    - Dictionary with 'HDD_price' and 'CDD_price'
    """
```

## 📚 Academic Foundation

This project is heavily based on the academic work:
- **Book**: "Weather derivatives modeling and pricing weather-related risk"
- **Authors**: Antonis K. Alexandridis and Achilleas Zapranis
- **Focus**: Mathematical modeling and pricing of weather-related financial instruments

## 🛠️ Technical Stack

- **Python 3.8+**
- **Data Science**: pandas, numpy, scipy
- **Time Series**: statsmodels, arch
- **Machine Learning**: PyTorch, scikit-learn
- **Signal Processing**: PyWavelets
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: Hurst exponent, ADF tests

## 📋 Dependencies

```
pandas
numpy
statsmodels
pywt
torch
scipy
matplotlib
hurst
ipywidgets
scikit-learn
```

## 🎯 Key Features

- **Multi-Model Comparison**: Three different forecasting approaches
- **Confidence Intervals**: Risk assessment through prediction intervals
- **Reference Temperature**: Historical data-based calculation
- **Performance Metrics**: Comprehensive model evaluation
- **Visualization**: Extensive plotting and analysis tools
- **Academic Rigor**: Based on established quantitative finance literature

## 📊 Output Files

- **Forecasts**: CSV files with model predictions and confidence intervals
- **Plots**: PNG files showing ACF/PACF analysis and forecast comparisons
- **Results**: JSON files with Monte Carlo simulation results

## 🤝 Contributing

This project was developed for the Minerva Student Association. For contributions or questions, please contact the project maintainers.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Minerva Student Association for project support
- Antonis K. Alexandridis and Achilleas Zapranis for the foundational academic work
- The quantitative finance community for weather derivatives research

---

*This project represents a comprehensive implementation of weather derivatives pricing methodologies, combining traditional time series analysis with modern machine learning and advanced stochastic modeling techniques.*
