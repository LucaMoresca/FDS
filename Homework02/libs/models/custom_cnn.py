import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################  
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Assuming input size 32x32
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # For 10 classes (e.g., CIFAR-10)

        # Activation function
        self.relu = nn.ReLU()
        
        pass

    def forward(self, x):
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################  
        # Define the forward pass
        x = self.relu(self.conv1(x))  # First convolution + ReLU
        x = self.pool(x)             # Pooling
        x = self.relu(self.conv2(x))  # Second convolution + ReLU
        x = self.pool(x)             # Pooling again
        x = x.view(x.size(0), -1)    # Flatten
        x = self.relu(self.fc1(x))   # Fully connected layer + ReLU
        x = self.fc2(x)              # Output layer
        return x