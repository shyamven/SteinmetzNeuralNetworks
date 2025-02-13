import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from utils.process_data import fft_transform

def LoadDataset(DatasetName):
    if DatasetName == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        
        # Small sample MNIST test case
        # X_train, y_train = X_train[:500,:,:,:], y_train[:500]
        
        # Transform data to complex domain using FFT
        X_train_real, X_train_imag = fft_transform(X_train)
        X_test_real, X_test_imag = fft_transform(X_test)
        
        print("MNIST dataset loaded")
        
    elif DatasetName == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.CIFAR10(root="data/", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="data/", train=False, transform=transform, download=True)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        
        # Transform data to complex domain using FFT
        X_train_real, X_train_imag = fft_transform(X_train)
        X_test_real, X_test_imag = fft_transform(X_test)
        
        print("CIFAR10 dataset loaded")
        
    elif DatasetName == "CIFAR100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.CIFAR100(root="data/", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root="data/", train=False, transform=transform, download=True)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        
        # Transform data to complex domain using FFT
        X_train_real, X_train_imag = fft_transform(X_train)
        X_test_real, X_test_imag = fft_transform(X_test)
        
        print("CIFAR100 dataset loaded")
    
    elif DatasetName == "ChannelID":
        s_n, r_tilde_n = generate_channel_data()
        X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test = create_dataloader(s_n, r_tilde_n)
        y_train = y_train.unsqueeze(1)
        y_test = y_test.unsqueeze(1)
        
        print("ChannelID dataset loaded")
        
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
  
    return X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test

def generate_channel_data(num_samples=2000, rho=np.sqrt(2)/2, snr_db=5):
    # Generate input signals s_n
    X_n = np.random.randn(num_samples)
    Y_n = np.random.randn(num_samples)
    s_n = np.sqrt(1 - rho**2) * X_n + 1j * rho * Y_n

    # Define the filter coefficients h(k)
    h = np.array([0.432 * (1 + np.cos(2 * np.pi * (k - 3) / 5) - 1j * (1 + np.cos(2 * np.pi * (k - 3) / 10)))
                  for k in range(1, 6)])

    # Apply the linear filter
    t_n = np.convolve(s_n, h, mode='full')[:num_samples]

    # Apply the memoryless nonlinearity
    r_n = t_n + (0.15 - 0.1j) * t_n**2

    # Add white Gaussian noise to get the final signal
    signal_power = np.mean(np.abs(r_n)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    r_tilde_n = r_n + noise

    return s_n, r_tilde_n

def create_dataloader(s_n, r_tilde_n, test_size=0.5):
    L = 5
    X = np.array([s_n[i:i+L] for i in range(len(s_n) - L)])
    y = r_tilde_n[L:]

    num_train = int((1 - test_size) * len(X))
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]

    X_train_real, X_train_imag = torch.tensor(X_train.real, dtype=torch.float32), torch.tensor(X_train.imag, dtype=torch.float32)
    X_test_real, X_test_imag = torch.tensor(X_test.real, dtype=torch.float32), torch.tensor(X_test.imag, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.complex64), torch.tensor(y_test, dtype=torch.complex64)

    return X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test
