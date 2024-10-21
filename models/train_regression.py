import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.classes import SteinmetzNetwork, NeuralNetwork, ComplexNeuralNetwork
from utils.process_data import add_complex_normal_noise, fft_based_hilbert_transform

def train_regression(X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test, task, model_choice, iterations, epochs, batch_size, noise):
    # Training loop
    error_mag = np.zeros((iterations, epochs))
    error_phase = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        if noise >= 0:
            # If noise mean > 0, add complex normal noise
            X_train_real, X_train_imag = add_complex_normal_noise(X_train_real, X_train_imag, noise)
        
        # Model instantiation based on choice
        input_size = X_train_real.shape[1]
        num_dims = y_train.shape[1]
        latent_size = 64
        
        if model_choice == "RVNN":
            model = NeuralNetwork(input_size, num_dims*2, latent_size)
        elif model_choice == "CVNN":
            model = ComplexNeuralNetwork(input_size, num_dims, latent_size)
        elif model_choice == "Steinmetz":
            model = SteinmetzNetwork(input_size, num_dims*2, latent_size)
        elif model_choice == "Analytic":
            model = SteinmetzNetwork(input_size, num_dims*2, latent_size)
        else:
            print("Error: Model choice not recognized.")
            exit(1)
        
        # Training neural network
        criterion = nn.MSELoss()
        learning_rate = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(sum(p.numel() for p in model.parameters()))
        mag_error, phase_error = train(X_train_real, X_train_imag, X_test_real, X_test_imag, y_train, y_test, iterations, i, epochs, batch_size, model, optimizer, criterion, model_choice, task)
            
        error_mag[i,:] = mag_error
        error_phase[i,:] = phase_error

    return error_mag, error_phase


def calculate_mag_phase_error(outputs, targets, criterion):
    # Calculate magnitude and phase of outputs and targets
    mag_outputs = torch.sqrt(outputs[:, 0]**2 + outputs[:, 1]**2) if outputs.shape[1] == 2 else torch.abs(outputs)
    phase_outputs = torch.atan2(outputs[:, 1], outputs[:, 0]) if outputs.shape[1] == 2 else torch.angle(outputs)
    mag_targets = torch.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
    phase_targets = torch.atan2(targets[:, 1], targets[:, 0])
    
    # Calculate magnitude and phase errors using the given criterion
    mag_error = criterion(mag_outputs, mag_targets).item()
    phase_error = criterion(phase_outputs, phase_targets).item()
    
    return mag_error, phase_error


def train(X_train_tensor_real, X_train_tensor_imag, X_test_tensor_real, X_test_tensor_imag, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, model_choice, task):
    # Training and test datasets
    y_train_tensor = torch.cat((y_train_tensor.real, y_train_tensor.imag), dim=1)
    y_test_tensor = torch.cat((y_test_tensor.real, y_test_tensor.imag), dim=1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor_real, X_train_tensor_imag, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor_real, X_test_tensor_imag, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Instantiate train and test loss
    mag_error = np.zeros(epochs)
    phase_error = np.zeros(epochs)
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        for real, imag, target in train_loader:
            optimizer.zero_grad()
            if model_choice == 'Steinmetz':
                outputs, _, _ = model(real, imag, model_choice)
                loss = criterion(outputs, target)
            elif model_choice == 'Analytic':
                outputs, real_features, imag_features = model(real, imag, model_choice)
                loss = custom_loss(outputs, target, real_features, imag_features)
            elif model_choice == 'RVNN':
                outputs = model(real, imag, task)
                loss = criterion(outputs, target)
            else:
                outputs = model(real, imag, task)
                loss = criterion(torch.cat((outputs.real, outputs.imag), dim=1), target)
            loss.backward()
            optimizer.step()
            
        # Validation loop
        model.eval()
        mag_err_sum = 0
        phase_err_sum = 0
        count = 0
        
        with torch.no_grad():
            for real, imag, target in test_loader:
                if model_choice == 'Steinmetz' or model_choice == 'Analytic':
                    outputs, _, _ = model(real, imag, model_choice)
                else:
                    outputs = model(real, imag, task)
                
                mag_err, phase_err = calculate_mag_phase_error(outputs, target, criterion)
                mag_err_sum += mag_err * real.size(0)
                phase_err_sum += phase_err * real.size(0)
                count += real.size(0)

        mag_error[epoch] = mag_err_sum / count
        phase_error[epoch] = phase_err_sum / count

        print(f'Iter [{i+1}/{iterations}], Epoch [{epoch+1}/{epochs}], {model_choice} Magnitude Error: {mag_error[epoch]:.4f}, Phase Error: {phase_error[epoch]:.4f}')
        
    return mag_error, phase_error


# Custom loss function for Analytic Neural Network
def custom_loss(outputs, target, real_features, imag_features):
    # Implementing hilbert consistency penalty + custom loss function
    transformed_imag = fft_based_hilbert_transform(real_features)
    consistency_penalty = nn.functional.mse_loss(transformed_imag, imag_features)
    beta = 5e-5 # tradeoff parameter

    return nn.MSELoss()(outputs, target) + beta*consistency_penalty