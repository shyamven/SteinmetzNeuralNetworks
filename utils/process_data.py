import numpy as np
import torch
        
def train_test_split_fixed(X, y, train_size, shuffle=True, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    - X, y: Arrays or matrices.
    - train_size: int, size of training dataset.
    - shuffle: Whether or not to shuffle the data before splitting.
    - random_state: Seed for reproducibility.
    
    Returns:
    - Split data into X_train, X_test, y_train, y_test.
    """
    
    # Ensure X and y have the same number of samples
    assert X.shape[0] == y.shape[0], "Inconsistent number of samples between X and y."
    
    # Ensure train_size is valid
    if not (0 <= train_size < X.shape[0]):
        print("Error: TrainSize must be between 0 and the total number of samples.")
        exit(1)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(X.shape[0])
    else:
        indices = np.arange(X.shape[0])
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def add_complex_normal_noise(X_real, X_imag, beta):
    """
    Add complex normal noise with specified scaling factor (beta) to complex data (X_real + i*X_imag).
    
    Parameters:
    - X_real (torch.Tensor): The input real part of the data.
    - X_imag (torch.Tensor): The input imaginary part of the data.
    - beta (float): The scaling factor.
    
    Returns:
    - noisy_X_real (torch.Tensor): The noisy real part of the data.
    - noisy_X_imag (torch.Tensor): The noisy imaginary part of the data.
    """
    
    # Generate Gaussian noise of the same shape as the input data
    noise_real = torch.randn_like(X_real) * beta / torch.sqrt(torch.tensor(2.0))
    noise_imag = torch.randn_like(X_imag) * beta / torch.sqrt(torch.tensor(2.0))
    
    # Add the noise to the real and imaginary parts of the input data
    noisy_X_real = X_real + noise_real
    noisy_X_imag = X_imag + noise_imag
    
    return noisy_X_real, noisy_X_imag


def fft_transform(data):
    """
    Compute FFT of data
    ----------
    Parameters:
    - data (pytorch tensor): The input data

    Returns
    - real and imaginary components (pytorch tensors) after FFT
    """
    data_fft = torch.fft.fft(data.view(data.size(0), -1), dim=1)
    real_data = data_fft.real
    imag_data = data_fft.imag
    return real_data, imag_data


def fft_based_hilbert_transform(real_features):
    """
    Apply the FFT-based Hilbert transform to the real features to obtain the imaginary part.

    Parameters:
    - real_features (torch.Tensor): The input real features.

    Returns:
    - transformed_imag (torch.Tensor): The transformed imaginary features obtained from the Hilbert transform.
    """
    # Perform FFT
    fft_result = torch.fft.fft(real_features, dim=-1)

    # Get the number of samples and create a tensor to hold the phase shifts
    N = real_features.shape[-1]
    phase_shift = torch.zeros_like(fft_result)

    # Apply a -90 degree phase shift for positive frequencies (1 to N/2 - 1)
    # and a +90 degree phase shift for negative frequencies (N/2 + 1 to N - 1)
    if N % 2 == 0:
        # Even number of samples
        phase_shift[..., 1:N//2] = -1j  # Positive frequencies (excluding Nyquist)
        phase_shift[..., N//2+1:] = 1j  # Negative frequencies
    else:
        # Odd number of samples
        phase_shift[..., 1:(N+1)//2] = -1j  # Positive frequencies
        phase_shift[..., (N+1)//2:] = 1j   # Negative frequencies

    # Apply phase shift and perorm inverse FFT
    shifted_fft_result = fft_result * phase_shift
    transformed_imag = torch.fft.ifft(shifted_fft_result, dim=-1).real

    return transformed_imag
