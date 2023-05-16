# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    print("mean: ", np.mean(window, axis=0))
    return np.mean(window, axis=0)
    

def _compute_entropy_features(window):
    """
    Computes the entropy of the x, y and z acceleration over the given window. 
    """
    mag = np.sqrt(np.sum(window**2))
    # Compute the histogram of accelerometer values and use it to find the discrete distribution
    hist, bins = np.histogram(mag, bins=10, density=True)
    p = hist * np.diff(bins)
    p = np.where(p == 0, np.finfo(float).eps, p) # Add small constant where p is zero
    entropy = -np.sum(p * np.log2(p))
    # print("entropy: ", entropy)

    # Compute the average entropy
    return entropy

def _compute_fft_features(window):
    """
    Computes the fft of the x, y and z acceleration over the given window by using the magnitude of the acceleration.
    """
    # compute magnitude of acceleration
    mag = np.sqrt(np.sum(window**2))
    mag2 = np.linalg.norm(window)
    print("window: ")
    print(window)
    print("mag: ")
    print(mag2)
    print(mag2.shape)
    print(mag2.dtype)

    # Compute the real-valued FFT of the signal
    fft = np.fft.rfft(window)

    # Compute the absolute value of the FFT
    abs_fft = np.abs(fft)

    # Compute the dominant frequency index
    index = np.argmax(abs_fft)
 
    return abs_fft[index]

def _compute_peak_count(window):
    """
    Computes the peak count of the magnitude acceleration over the given window. 
    """
    # compute magnitude of acceleration
    mag = np.sqrt(np.sum(window**2))

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(window)

    # peak count in the window
    peak_count = len(peaks)

    # print("peak count: ", peak_count)

    return peak_count


def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    print(window)
    win = np.array(window)
    print("extracting")
    # print(_compute_mean_features(win[:,0]))
    # x.append(_compute_mean_features(win[:,0]))
    # feature_names.append("x_mean")

    # x.append(_compute_mean_features(win[:,1]))
    # feature_names.append("y_mean")

    # x.append(_compute_mean_features(win[:,2]))
    # feature_names.append("z_mean")

    x.append(_compute_entropy_features(win))
    feature_names.append("average_entropy")

    x.append(_compute_fft_features(win))
    feature_names.append("dominant_freq")

    x.append(_compute_peak_count(win))
    feature_names.append("peak_count")

    feature_vector = list(x)
    return feature_names, feature_vector