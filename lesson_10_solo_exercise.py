# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:37:18 2025

@author: taske
"""

import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.datasets import load_digits  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import numpy as np  
  
# 1. TODO: Load and preprocess your dataset  
digits = load_digits()
X = digits.data
y = digits.target
  
# 2. TODO: Split into training and test sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  
# 3. TODO: Convert your splits to PyTorch tensors  
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)
  
# 4. TODO: Define a simple PyTorch model  
# class SimpleNet(nn.Module):  
#     def __init__(self, input_dim, output_dim):  
#         super(SimpleNet, self).__init__()  
#         # layers here  
#     def forward(self, x):  
#         # forward pass here  
  
# input_dim = ...  
# output_dim = ...  
# model = SimpleNet(input_dim, output_dim)  
  
# 5. TODO: Specify loss function and optimizer  
# criterion = ...  
# optimizer = ...  
  
# 6. TODO: Training loop  
num_epochs = 20  
batch_size = 32  
loss_history = []  
  
# for epoch in range(num_epochs):  
#     # Shuffle and batch your data  
#     # Forward, backward, optimizer.step()  
#     # Track and store average loss in loss_history  
  
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  
  
# 7. TODO: Evaluation (accuracy, etc.)  
# with torch.no_grad():  
#     outputs = ...  
#     predicted = ...  
#     accuracy = ...  
#     print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")  
  
# 8. Plotting (READY, but will fail until above done)  
plt.figure()  
plt.plot(range(1, num_epochs+1), loss_history, marker='o')  
plt.xlabel("Epoch")  
plt.ylabel("Loss")  
plt.title("Training Loss Curve")  
plt.grid()  
plt.tight_layout()  
plt.savefig('loss_curve.png')  
print("Loss curve saved as loss_curve.png")