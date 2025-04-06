#NOTE wavelet analysis for seasonal mean estimation

import pywt
import numpy as np

def remove_seasonality(temp_data):
    wavelet = 'db11'
    level = 11
    coefs = pywt.wavedec(temp_data, wavelet, level=level)

    # Zero out coefficients outside seasonal levels
    seasonal_coefs = []
    for i, c in enumerate(coefs):
        # Keep detail levels d6–d10 as seasonal, set others to zero
        if i >= 1 and i <= 5:  # d1–d5 (high frequency noise)
            seasonal_coefs.append(np.zeros_like(c))
        else:
            seasonal_coefs.append(c)

    seasonal_component = pywt.waverec(seasonal_coefs, wavelet)
    deseasoned_data = temp_data - seasonal_component
    return deseasoned_data


