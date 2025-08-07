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
class SimpleNet(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(SimpleNet, self).__init__()
         self.Linear1 = torch.nn.Linear(input_dim, 16)
         self.ReLU1 = torch.nn.ReLU()
         self.Dropout1 = torch.nn.Dropout(0.5)
         self.Linear2 = torch.nn.Linear(16,32)
         self.ReLU2 = torch.nn.ReLU()
         self.Dropout2 = torch.nn.Dropout(0.5)
         self.Linear3 = torch.nn.Linear(32,16)
         self.ReLU3 = torch.nn.ReLU()
         self.Dropout3 = torch.nn.Dropout(0.5)
         self.Linear4 = torch.nn.Linear(16,8)
         self.ReLU4 = torch.nn.ReLU()
         self.Dropout4 = torch.nn.Dropout(0.5)
         self.Linear5 = torch.nn.Linear(8,output_dim)
         self.Softmax = torch.nn.Softmax(dim=1)

     def forward(self, x):
         x = self.Linear1(x)
         x = self.ReLU1(x)
         x = self.Dropout1(x)
         x = self.Linear2(x)
         x = self.ReLU2(x)
         x = self.Dropout2(x)
         x = self.Linear3(x)
         x = self.ReLU3(x)
         x = self.Dropout3(x)
         x = self.Linear4(x)
         x = self.ReLU4(x)
         x = self.Dropout4(x)
         x = self.Linear5(x)
         x = self.Softmax(x)
         return x
input_dim = digits.data.shape[1]
output_dim = digits.target.shape[1]
model = SimpleNet(input_dim, output_dim)
  
# 5. TODO: Specify loss function and optimizer  
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
# 6. TODO: Training loop  
num_epochs = 20  
batch_size = 32  
loss_history = []  
  
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

  
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {(loss/len(X_train)).item():.4f}")
  
# 7. TODO: Evaluation (accuracy, etc.)  
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = torch.mean(predicted == y_test_tensor)

    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
  
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