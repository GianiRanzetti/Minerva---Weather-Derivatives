#NOTE wavelet analysis for seasonal mean estimation

import pywt
import numpy as np

def wavelet_decompose(signal, wavelet_name='db11', max_level=None):
    """
    Decomposes a signal using Discrete Wavelet Transform (DWT)
    and returns a standard-order coefficient dictionary.

    Parameters:
        signal (array-like): The time series to decompose.
        wavelet_name (str): The name of the wavelet (default: 'db11').
        max_level (int): Max level of decomposition. If None, it is auto-calculated.

    Returns:
        dict: Dictionary with keys:
              'approx' (A_n) and 'detail_1' (D_1, highest freq), ..., 'detail_n' (D_n, lowest freq)
    """
    wavelet = pywt.Wavelet(wavelet_name)
    
    if max_level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=max_level)
    
    # Reverse the order of detail coefficients to match standard D_1 (highest freq), D_n (lowest)
    detail_coeffs = coeffs[1:][::-1]
    
    coeff_dict = {'approx': coeffs[0]}
    for i, d in enumerate(detail_coeffs, start=1):
        coeff_dict[f'detail_{i}'] = d
    
    return coeff_dict


def wavelet_reconstruct(coeff_dict, keep_levels, wavelet_name='db11'):
    """
    Reconstructs a signal from selected wavelet levels (standard-labeled).
    """
    # Determine how many detail levels we have
    num_details = len(coeff_dict) - 1  # excludes 'approx'

    # Reconstruct the coeffs list in pywt format: [A_n, D_n, ..., D_1]
    coeffs = [coeff_dict['approx']]
    
    # Reverse back to match pywt's expected [D_n, ..., D_1]
    for i in range(num_details, 0, -1):
        key = f'detail_{i}'
        if key in keep_levels:
            coeffs.append(coeff_dict[key])
        else:
            coeffs.append(np.zeros_like(coeff_dict[key]))

    return pywt.waverec(coeffs, wavelet_name)