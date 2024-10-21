import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

# Define the neural network models
class SteinmetzNetwork(nn.Module):
    def __init__(self, dN, k, lN):
        super(SteinmetzNetwork, self).__init__()
        self.real_net = nn.Sequential(nn.Linear(dN, lN//2), nn.ReLU(), nn.Linear(lN//2, lN//2), nn.ReLU())
        self.imag_net = nn.Sequential(nn.Linear(dN, lN//2), nn.ReLU(), nn.Linear(lN//2, lN//2), nn.ReLU())
        self.regressor = nn.Sequential(nn.Linear(lN, k))

    def forward(self, real, imag, model_choice):
        real_features = self.real_net(real)
        imag_features = self.imag_net(imag)
        
        # Mean centering features as last step before concatenation
        # real_features = real_features - real_features.mean(dim=0, keepdim=True)
        imag_features = imag_features - imag_features.mean(dim=0, keepdim=True)
        
        combined = torch.cat((real_features, imag_features), dim=1)
        output = self.regressor(combined)
        return output, real_features, imag_features

class NeuralNetwork(nn.Module):
    def __init__(self, dN, k, lN):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(2*dN, lN//2), nn.ReLU(), nn.Linear(lN//2, lN), nn.ReLU(), nn.Linear(lN, k))

    def forward(self, real, imag, task):
        input = torch.cat((real, imag), dim=1)
        output = self.net(input)
        return output

class ComplexNeuralNetwork(nn.Module):
    def __init__(self, dN, k, lN):
        super(ComplexNeuralNetwork, self).__init__()
        self.fc1 = ComplexLinear(dN, lN//2)
        self.fc2 = ComplexLinear(lN//2, lN)
        self.fc3 = ComplexLinear(lN, k)

    def forward(self, real, imag, task):
        complex_tensor = torch.stack((real, imag), dim=-1)
        x = torch.view_as_complex(complex_tensor)
        x = complex_relu(self.fc1(x))
        x = complex_relu(self.fc2(x))
        x = self.fc3(x)
        if task == 'Classification':
            output = torch.abs(x)
        elif task == 'Regression':
            output = x
        return output