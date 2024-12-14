#!/usr/bin/env python
# coding: utf-8

# In[44]:


get_ipython().system('jupyter nbconvert --to script Assignment5.ipynb')


# In[2]:


# Question 2a #

import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1
y_train = np.where(y_train == 0, -1, 1).reshape(-1, 1)
y_test = np.where(y_test == 0, -1, 1).reshape(-1, 1)

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights
def initialize_weights(input_dim, hidden_dim, output_dim):
    return {
        "w1": np.random.normal(0, 1, (hidden_dim, input_dim + 1)),  # Including bias
        "w2": np.random.normal(0, 1, (hidden_dim, hidden_dim + 1)),
        "w3": np.random.normal(0, 1, (output_dim, hidden_dim + 1))
    }

# Forward pass
def forward_pass(X, weights):
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias to input
    z1 = sigmoid(np.dot(X_with_bias, weights["w1"].T))
    z1_with_bias = np.hstack((np.ones((z1.shape[0], 1)), z1))  # Add bias to layer 1
    z2 = sigmoid(np.dot(z1_with_bias, weights["w2"].T))
    z2_with_bias = np.hstack((np.ones((z2.shape[0], 1)), z2))  # Add bias to layer 2
    y_pred = np.dot(z2_with_bias, weights["w3"].T)  # Output layer
    return X_with_bias, z1_with_bias, z2_with_bias, y_pred

# Backward pass
def backward_pass(X_with_bias, z1_with_bias, z2_with_bias, y_pred, y_true, weights, learning_rate):
    m = y_true.shape[0]
    delta3 = (y_pred - y_true) / m  # Output error term
    grad_w3 = np.dot(delta3.T, z2_with_bias)

    delta2 = np.dot(delta3, weights["w3"][:, 1:]) * sigmoid_derivative(z2_with_bias[:, 1:])
    grad_w2 = np.dot(delta2.T, z1_with_bias)

    delta1 = np.dot(delta2, weights["w2"][:, 1:]) * sigmoid_derivative(z1_with_bias[:, 1:])
    grad_w1 = np.dot(delta1.T, X_with_bias)

    # Update weights
    weights["w3"] -= learning_rate * grad_w3
    weights["w2"] -= learning_rate * grad_w2
    weights["w1"] -= learning_rate * grad_w1

    return weights

# Train the neural network
def train_neural_network(X_train, y_train, hidden_dim, output_dim, learning_rate, epochs):
    input_dim = X_train.shape[1]
    weights = initialize_weights(input_dim, hidden_dim, output_dim)
    for epoch in range(epochs):
        X_with_bias, z1_with_bias, z2_with_bias, y_pred = forward_pass(X_train, weights)
        weights = backward_pass(X_with_bias, z1_with_bias, z2_with_bias, y_pred, y_train, weights, learning_rate)
    return weights

# Predict
def predict(X, weights):
    _, _, _, y_pred = forward_pass(X, weights)
    return np.sign(y_pred)

# Hyperparameters
hidden_dim = 10  
output_dim = 1  
learning_rate = 0.01
epochs = 100

# Train the neural network
weights = train_neural_network(X_train, y_train, hidden_dim, output_dim, learning_rate, epochs)

# Evaluate the model
y_pred_train = predict(X_train, weights)
y_pred_test = predict(X_test, weights)

train_error = np.mean(y_pred_train != y_train)
test_error = np.mean(y_pred_test != y_test)

print(f"Training Error: {train_error}")
print(f"Test Error: {test_error}")


# In[50]:


# Question 2b #


import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

# Convert labels
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

class NeuralNetwork:
    def __init__(self, width, input_dim):
        self.width = width
        self.input_dim = input_dim
        self.weights = self.initialize_weights()
        
    def initialize_weights(self):
        #"""Initialize weights from standard Gaussian distribution"""
        return {
            "w1": np.random.normal(0, 1, (self.width, self.input_dim + 1)),
            "w2": np.random.normal(0, 1, (self.width, self.width + 1)),
            "w3": np.random.normal(0, 1, (1, self.width + 1))
        }
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # First layer
        z1 = np.dot(X_bias, self.weights["w1"].T)
        a1 = self.sigmoid(z1)
        a1_bias = np.hstack((np.ones((a1.shape[0], 1)), a1))
        
        # Second layer
        z2 = np.dot(a1_bias, self.weights["w2"].T)
        a2 = self.sigmoid(z2)
        a2_bias = np.hstack((np.ones((a2.shape[0], 1)), a2))
        
        # Output layer
        y_pred = np.dot(a2_bias, self.weights["w3"].T)
        
        return X_bias, z1, a1_bias, z2, a2_bias, y_pred
    
    def compute_loss(self, y_pred, y_true):
        #"""Compute squared loss"""
        return 0.5 * np.mean((y_pred - y_true) ** 2)

def get_learning_rate(gamma0, d, t):
    #"""Implement the learning rate schedule"""
    return gamma0 / (1 + (gamma0 * t / d))

def train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d, epochs=100):
    #"""Train network and return errors"""
    nn = NeuralNetwork(width=width, input_dim=X_train.shape[1])
    t = 0  
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(len(X_train)):
            
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Forward pass
            X_bias, z1, a1_bias, z2, a2_bias, y_pred = nn.forward_pass(x_i)
            
            # Backward pass
            delta3 = (y_pred - y_i)
            grad_w3 = np.dot(delta3.T, a2_bias)
            
            # Second layer
            delta2 = np.dot(delta3, nn.weights["w3"][:, 1:]) * nn.sigmoid_derivative(z2)
            grad_w2 = np.dot(delta2.T, a1_bias)
            
            # First layer
            delta1 = np.dot(delta2, nn.weights["w2"][:, 1:]) * nn.sigmoid_derivative(z1)
            grad_w1 = np.dot(delta1.T, X_bias)
            
            # Update weights
            lr = get_learning_rate(gamma0, d, t)
            nn.weights["w3"] -= lr * grad_w3
            nn.weights["w2"] -= lr * grad_w2
            nn.weights["w1"] -= lr * grad_w1
            
            t += 1
    
    # Final errors
    _, _, _, _, _, y_pred_train = nn.forward_pass(X_train)
    _, _, _, _, _, y_pred_test = nn.forward_pass(X_test)
    
    train_error = np.mean(np.sign(y_pred_train) != y_train)
    test_error = np.mean(np.sign(y_pred_test) != y_test)
    
    return train_error, test_error

# Parameters
widths = [5, 10, 25, 50, 100]
gamma0_values = [0.1, 0.01]
d_values = [1, 10, 100]

print("\nResults for each width:")
print("Width\tgamma0\td\tTrain Error\tTest Error")
for width in widths:
    for gamma0 in gamma0_values:
        for d in d_values:
            train_error, test_error = train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d)
            print(f"{width}\t{gamma0:.3f}\t{d}\t{train_error:.4f}\t{test_error:.4f}")


# In[5]:


#Question 2C


import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

# Convert labels
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

class NeuralNetwork:
    def __init__(self, width, input_dim):
        self.width = width
        self.input_dim = input_dim
        self.weights = self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize all weights to zero
        return {
            "w1": np.zeros((self.width, self.input_dim + 1)),
            "w2": np.zeros((self.width, self.width + 1)),
            "w3": np.zeros((1, self.width + 1))
        }
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # First layer
        z1 = np.dot(X_bias, self.weights["w1"].T)
        a1 = self.sigmoid(z1)
        a1_bias = np.hstack((np.ones((a1.shape[0], 1)), a1))
        
        # Second layer
        z2 = np.dot(a1_bias, self.weights["w2"].T)
        a2 = self.sigmoid(z2)
        a2_bias = np.hstack((np.ones((a2.shape[0], 1)), a2))
        
        # Output layer
        y_pred = np.dot(a2_bias, self.weights["w3"].T)
        
        return X_bias, z1, a1_bias, z2, a2_bias, y_pred

def get_learning_rate(gamma0, d, t):
    return gamma0 / (1 + (gamma0 * t / d))

def train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d, epochs=100):
    nn = NeuralNetwork(width=width, input_dim=X_train.shape[1])
    t = 0  
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(len(X_train)):
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Forward pass
            X_bias, z1, a1_bias, z2, a2_bias, y_pred = nn.forward_pass(x_i)
            
            # Backward pass
            delta3 = (y_pred - y_i)
            grad_w3 = np.dot(delta3.T, a2_bias)
            
            # Second layer
            delta2 = np.dot(delta3, nn.weights["w3"][:, 1:]) * nn.sigmoid_derivative(z2)
            grad_w2 = np.dot(delta2.T, a1_bias)
            
            # First layer
            delta1 = np.dot(delta2, nn.weights["w2"][:, 1:]) * nn.sigmoid_derivative(z1)
            grad_w1 = np.dot(delta1.T, X_bias)
            
            # Update weights
            lr = get_learning_rate(gamma0, d, t)
            nn.weights["w3"] -= lr * grad_w3
            nn.weights["w2"] -= lr * grad_w2
            nn.weights["w1"] -= lr * grad_w1
            
            t += 1
    
    # Final errors
    _, _, _, _, _, y_pred_train = nn.forward_pass(X_train)
    _, _, _, _, _, y_pred_test = nn.forward_pass(X_test)
    
    train_error = np.mean(np.sign(y_pred_train) != y_train)
    test_error = np.mean(np.sign(y_pred_test) != y_test)
    
    return train_error, test_error

# Test with different widths
widths = [5, 10, 25, 50, 100]

gamma0 = 0.01
d = 1

print("\nResults for Zero Initialization:")
print("Width\tTrain Error\tTest Error")
for width in widths:
    train_error, test_error = train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d)
    print(f"{width}\t{train_error:.4f}\t\t{test_error:.4f}")
    


# In[13]:


# Question 2E

#!pip install torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Step 1: Load Data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    
    X_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)
    
    # Convert labels: 0 -> -1, 1 -> 1
    y_train = torch.where(y_train == 0, torch.tensor(-1.0), torch.tensor(1.0))
    y_test = torch.where(y_test == 0, torch.tensor(-1.0), torch.tensor(1.0))
    
    return X_train, y_train, X_test, y_test

# Step 2: Define the Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, width):  # Fixed initialization syntax
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Define the Loss Function and Optimizer
def train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d, epochs=100, batch_size=32):
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim, width)
    
    # Changed to BCEWithLogitsLoss 
    criterion = nn.BCEWithLogitsLoss()
    
    # Implemented learning rate schedule
    lr_lambda = lambda epoch: gamma0 / (1 + gamma0/d * epoch)
    optimizer = optim.SGD(model.parameters(), lr=gamma0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Step 4: Prepare DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Convert labels from [-1, 1] to [0, 1] for BCEWithLogitsLoss
            batch_y_binary = (batch_y + 1) / 2
            
            loss = criterion(outputs, batch_y_binary)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Step 5: Evaluate Model
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train)
        test_predictions = model(X_test)
        
        # Convert sigmoid outputs to [-1, 1] predictions
        train_predictions = torch.where(train_predictions >= 0, 
                                      torch.tensor(1.0), 
                                      torch.tensor(-1.0))
        test_predictions = torch.where(test_predictions >= 0, 
                                     torch.tensor(1.0), 
                                     torch.tensor(-1.0))
    
    train_error = (train_predictions != y_train).float().mean().item()
    test_error = (test_predictions != y_test).float().mean().item()
    
    return train_error, test_error

# Step 6: Hyperparameter Tuning
def main():
    train_path = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv'
    test_path = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv'
    
    try:
        X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    except FileNotFoundError:
        print("Error: Could not find the data files. Please check the file paths.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    widths = [5, 10, 25, 50, 100]
    gamma0_values = [0.1, 0.01]
    d_values = [1, 10, 100]

    print("Width\tGamma0\tD\tTrain Error\tTest Error")
    for width in widths:
        for gamma0 in gamma0_values:
            for d in d_values:
                try:
                    train_error, test_error = train_and_evaluate(
                        X_train, y_train, X_test, y_test, 
                        width, gamma0, d
                    )
                    print(f"{width}\t{gamma0}\t{d}\t{train_error:.4f}\t{test_error:.4f}")
                except Exception as e:
                    print(f"Error training model with width={width}, gamma0={gamma0}, d={d}: {str(e)}")

if __name__ == "__main__":
    main()
    


# In[27]:


# Question 3a #

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions)))
    return cost

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, variance, learning_rate=0.01, epochs=100):
    m, n = X.shape
    weights = np.random.normal(0, np.sqrt(variance), size=n)  # Initialize weights with given variance
    cost_history = []

    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(0, m)
            x_i = X[rand_index, :].reshape(1, -1)
            y_i = y[rand_index]
            prediction = sigmoid(np.dot(x_i, weights))
            gradient = np.dot(x_i.T, (prediction - y_i))
            weights -= learning_rate * gradient.flatten()

        # Compute cost for this epoch
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights

# Add bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Variance
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
learning_rate = 0.01
epochs = 100

# Store errors
results = []

for variance in variances:
    # Train logistic regression
    weights = stochastic_gradient_descent(X_train, y_train, variance, learning_rate, epochs)

    # Predict on train and test data
    def predict(X, weights):
        probabilities = sigmoid(np.dot(X, weights))
        return [1 if p >= 0.5 else 0 for p in probabilities]

    y_pred_train = predict(X_train, weights)
    y_pred_test = predict(X_test, weights)

    # Calculate errors
    train_error = 1 - accuracy_score(y_train, y_pred_train)
    test_error = 1 - accuracy_score(y_test, y_pred_test)

    results.append((variance, train_error, test_error))
    print(f"Variance: {variance} - Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

# Display results
print("\nResults Summary:")
print("Variance\tTrain Error\tTest Error")
for variance, train_error, test_error in results:
    print(f"{variance}\t\t{train_error:.4f}\t\t{test_error:.4f}")


print(f"\nFinal Results:")
print(f"Train Error: {final_train_error:.4f}")
print(f"Test Error: {final_test_error:.4f}")


# In[25]:


#Question 3b

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load training and test data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Add bias term to X (intercept)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions)))
    return cost

def get_learning_rate(gamma0, d, t):
    return gamma0 / (1 + (gamma0 * t / d))

def stochastic_gradient_descent_ml(X, y, gamma0, d, epochs=100):
    m, n = X.shape
    weights = np.zeros(n)  
    cost_history = []
    t = 0  
    
    for epoch in range(epochs):
        # Shuffle the training data at the start of each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i]
            
            # Get current learning rate
            learning_rate = get_learning_rate(gamma0, d, t)
            
            # Compute gradient and update weights
            prediction = sigmoid(np.dot(x_i, weights))
            gradient = np.dot(x_i.T, (prediction - y_i))
            weights -= learning_rate * gradient.flatten()
            
            # Increment update counter
            t += 1
            
            if t % 100 == 0:
                cost = compute_cost(X, y, weights)
                cost_history.append((t, cost))
    
    return weights, cost_history

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int)

# Try different combinations of gamma0 and d
gamma0_values = [0.1, 0.01, 0.001]
d_values = [1, 10, 100]
epochs = 100

results = []  
best_train_error = float('inf')
best_params = None
best_weights = None
best_cost_history = None

for gamma0 in gamma0_values:
    for d in d_values:
        print(f"\nTrying gamma0={gamma0}, d={d}")
        
        # Train model
        weights, cost_history = stochastic_gradient_descent_ml(X_train, y_train, gamma0, d, epochs)
        
        # Calculate training and test errors
        y_pred_train = predict(X_train, weights)
        y_pred_test = predict(X_test, weights)
        train_error = 1 - accuracy_score(y_train, y_pred_train)
        test_error = 1 - accuracy_score(y_test, y_pred_test)
        
        print(f"Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
        results.append((gamma0, d, train_error, test_error))
        
        # Track best parameters
        if train_error < best_train_error:
            best_train_error = train_error
            best_params = (gamma0, d)
            best_weights = weights
            best_cost_history = cost_history

# Print results for each combination
print("\nResults Summary for Each Combination:")
print("gamma0\td\tTrain Error\tTest Error")
for gamma0, d, train_error, test_error in results:
    print(f"{gamma0}\t{d}\t{train_error:.4f}\t\t{test_error:.4f}")

# Using best parameters for final result
print(f"\nBest parameters: gamma0={best_params[0]}, d={best_params[1]}")

# Final predictions 
y_pred_train = predict(X_train, best_weights)
y_pred_test = predict(X_test, best_weights)

# Calculate final errors
final_train_error = 1 - accuracy_score(y_train, y_pred_train)
final_test_error = 1 - accuracy_score(y_test, y_pred_test)

print(f"\nFinal Results:")
print(f"Train Error: {final_train_error:.4f}")
print(f"Test Error: {final_test_error:.4f}")


# In[52]:


# Question 3C

import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

# Convert labels 
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

class NeuralNetwork:
    def __init__(self, width, input_dim):
        self.width = width
        self.input_dim = input_dim
        self.weights = self.initialize_weights()
        
    def initialize_weights(self):
        return {
            "w1": np.zeros((self.width, self.input_dim + 1)),  # including bias
            "w2": np.zeros((self.width, self.width + 1)),
            "w3": np.zeros((1, self.width + 1))
        }
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # First hidden layer
        z1 = np.dot(X_bias, self.weights["w1"].T)
        a1 = self.sigmoid(z1)
        a1_bias = np.hstack((np.ones((a1.shape[0], 1)), a1))
        
        # Second hidden layer
        z2 = np.dot(a1_bias, self.weights["w2"].T)
        a2 = self.sigmoid(z2)
        a2_bias = np.hstack((np.ones((a2.shape[0], 1)), a2))
        
        # Output layer
        y_pred = np.dot(a2_bias, self.weights["w3"].T)
        
        return X_bias, z1, a1_bias, z2, a2_bias, y_pred
    
    def compute_loss(self, y_pred, y_true):
        """Compute squared loss"""
        return 0.5 * np.mean((y_pred - y_true) ** 2)

def get_learning_rate(gamma0, d, t):
    return gamma0 / (1 + (gamma0 * t / d))

def train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d, epochs=100):
    nn = NeuralNetwork(width=width, input_dim=X_train.shape[1])
    t = 0  
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(len(X_train)):
            # Get single sample
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Forward pass
            X_bias, z1, a1_bias, z2, a2_bias, y_pred = nn.forward_pass(x_i)
            
            # Backward pass
            delta3 = (y_pred - y_i)
            grad_w3 = np.dot(delta3.T, a2_bias)
            
            # Second hidden layer
            delta2 = np.dot(delta3, nn.weights["w3"][:, 1:]) * nn.sigmoid_derivative(z2)
            grad_w2 = np.dot(delta2.T, a1_bias)
            
            # First hidden layer
            delta1 = np.dot(delta2, nn.weights["w2"][:, 1:]) * nn.sigmoid_derivative(z1)
            grad_w1 = np.dot(delta1.T, X_bias)
            
            # Update weights
            lr = get_learning_rate(gamma0, d, t)
            nn.weights["w3"] -= lr * grad_w3
            nn.weights["w2"] -= lr * grad_w2
            nn.weights["w1"] -= lr * grad_w1
            
            t += 1
    
    # Calculating the errors
    _, _, _, _, _, y_pred_train = nn.forward_pass(X_train)
    _, _, _, _, _, y_pred_test = nn.forward_pass(X_test)
    
    train_error = np.mean(np.sign(y_pred_train) != y_train)
    test_error = np.mean(np.sign(y_pred_test) != y_test)
    
    return train_error, test_error

# Parameters to try
widths = [5, 10, 25, 50, 100]
gamma0_values = [0.1, 0.01, 0.001]
d_values = [1, 10, 100]

print("\nResults for each width and parameter setting:")
print("Width\tgamma0\td\tTrain Error\tTest Error")
for width in widths:
    for gamma0 in gamma0_values:
        for d in d_values:
            train_error, test_error = train_and_evaluate(X_train, y_train, X_test, y_test, width, gamma0, d)
            print(f"{width}\t{gamma0:.3f}\t{d}\t{train_error:.4f}\t{test_error:.4f}")


# In[ ]:




