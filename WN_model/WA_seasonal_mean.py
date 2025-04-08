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
    Reconstructs a signal from selected wavelet levels.

    Parameters:
        coeff_dict (dict): Dictionary of wavelet coefficients.
        keep_levels (list): Levels to keep (e.g., ['approx', 'detail_6', 'detail_7']).
        wavelet_name (str): Wavelet used for reconstruction.

    Returns:
        array: Reconstructed signal with only selected components.
    """
    coeffs = []
    max_level = len(coeff_dict) - 1
    
    # Create list of all possible keys
    all_keys = ['approx'] + [f'detail_{i}' for i in range(1, max_level + 1)]
    
    # Build coefficient list in the correct order
    for key in all_keys:
        if key in coeff_dict:
            if key in keep_levels:
                coeffs.append(coeff_dict[key])
            else:
                coeffs.append(np.zeros_like(coeff_dict[key]))
    
    return pywt.waverec(coeffs, wavelet_name)