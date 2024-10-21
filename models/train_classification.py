import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.classes import SteinmetzNetwork, NeuralNetwork, ComplexNeuralNetwork
from utils.process_data import add_complex_normal_noise, fft_based_hilbert_transform

def train_classification(X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test, task, model_choice, iterations, epochs, batch_size, noise):
    # Training loop
    acc_test = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        if noise >= 0:
            # If noise mean > 0, add complex normal noise
            X_train_real, X_train_imag = add_complex_normal_noise(X_train_real, X_train_imag, noise)
        
        # Model instantiation based on choice
        input_size = X_train_real.shape[1]
        num_classes = 10
        latent_size = 64
        
        if model_choice == "RVNN":
            model = NeuralNetwork(input_size, num_classes, latent_size)
        elif model_choice == "CVNN":
            model = ComplexNeuralNetwork(input_size, num_classes, latent_size)
        elif model_choice == "Steinmetz":
            model = SteinmetzNetwork(input_size, num_classes, latent_size)
        elif model_choice == "Analytic":
            model = SteinmetzNetwork(input_size, num_classes, latent_size)
        else:
            print("Error: Model choice not recognized.")
            exit(1)
        
        # Training neural network
        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-4  # MNIST = 1e-4, CIFAR10 = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(sum(p.numel() for p in model.parameters()))
        test_acc = train(X_train_real, X_train_imag, X_test_real, X_test_imag, y_train, y_test, iterations, i, epochs, batch_size, model, optimizer, criterion, model_choice, task)
            
        acc_test[i,:] = test_acc

    return acc_test

        
# Training Neural Network with Cross Entropy Loss
def train(X_train_tensor_real, X_train_tensor_imag, X_test_tensor_real, X_test_tensor_imag, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, model_choice, task):
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor_real, X_train_tensor_imag, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor_real, X_test_tensor_imag, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Instantiate train and test loss
    test_acc = np.zeros(epochs)
    
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
            else:
                outputs = model(real, imag, task)
                loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
        # Validation loop
        model.eval()
        correct = 0; total = 0;
        with torch.no_grad():
            for real, imag, target in test_loader:
                if model_choice == 'Steinmetz' or model_choice == 'Analytic':
                    outputs, _, _ = model(real, imag, model_choice)
                else:
                    outputs = model(real, imag, task)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_acc[epoch] = 100 * correct / total

        print(f'Iter [{i+1}/{iterations}], Epoch [{epoch+1}/{epochs}], {model_choice} Test Accuracy: {test_acc[epoch]:.2f}%')
        
    return test_acc


# Custom loss function for Analytic Neural Network
def custom_loss(outputs, target, real_features, imag_features):
    # Implementing hilbert consistency penalty + custom loss function
    transformed_imag = fft_based_hilbert_transform(real_features)
    consistency_penalty = nn.functional.mse_loss(transformed_imag, imag_features)
    beta = 1e-4 # tradeoff parameter

    return nn.CrossEntropyLoss()(outputs, target) + beta*consistency_penalty