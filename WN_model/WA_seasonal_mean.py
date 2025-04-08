#NOTE wavelet analysis for seasonal mean estimation

import pywt
import numpy as np
import pandas as pd

def remove_seasonality(temp_data):
    wavelet = 'db11'
    level = 11
    coefs = pywt.wavedec(temp_data, wavelet, level=level)

    # Zero out coefficients outside seasonal levels
    seasonal_coefs = []
    for i, c in enumerate(coefs):
        # Keep detail levels d6–d10 as seasonal, set others to zero
        if i >= 6 and i <= 10:  # Keep only d6-d10 (seasonal component)
            seasonal_coefs.append(c)
        else:
            seasonal_coefs.append(np.zeros_like(c))

    seasonal_component = pywt.waverec(seasonal_coefs, wavelet)
    
    # Ensure the output has the same length as the input
    if len(seasonal_component) > len(temp_data):
        seasonal_component = seasonal_component[:len(temp_data)]
    elif len(seasonal_component) < len(temp_data):
        # Pad with zeros if needed
        padding = np.zeros(len(temp_data) - len(seasonal_component))
        seasonal_component = np.concatenate([seasonal_component, padding])
    
    deseasoned_data = np.asarray(temp_data) - seasonal_component
    return pd.Series(deseasoned_data, index=temp_data.index)


