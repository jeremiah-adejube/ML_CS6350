#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().system('jupyter nbconvert --to script Assignment4.ipynb')


# In[14]:


# Question 2a #

import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features and labels for training and test datasets
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Implement SVM in the primal domain using stochastic sub-gradient descent
def svm_primal_sgd(X, y, C, epochs, gamma_0, a):
    weights = np.zeros(X.shape[1])  
    bias = 0  
    n_samples = len(y)

    for epoch in range(epochs):
       
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for t, (xi, target) in enumerate(zip(X, y)):
            learning_rate = gamma_0 / (1 + gamma_0 * t / a)  
            condition = target * (np.dot(weights, xi) + bias)

            if condition < 1:
                
                weights = (1 - learning_rate) * weights + learning_rate * C * target * xi
                bias += learning_rate * C * target
            else:
                
                weights = (1 - learning_rate) * weights

    return weights, bias

# Predict function for SVM
def predict_svm(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)

# Set hyperparameters
epochs = 100  
gamma_0 = 0.1  
a = 10.0  

# C values
C_values = [100/873, 500/873, 700/873]

# Train and test the model for C value
results = []
for C in C_values:
    weights, bias = svm_primal_sgd(X_train, y_train, C, epochs, gamma_0, a)
    y_pred_train = predict_svm(X_train, weights, bias)
    y_pred_test = predict_svm(X_test, weights, bias)
    
    # Errors
    train_error = np.mean(y_pred_train != y_train)
    test_error = np.mean(y_pred_test != y_test)
    
    results.append({
        'C': C,
        'Weights': weights,
        'Bias': bias,
        'Train Error': train_error,
        'Test Error': test_error
    })

# Results
for result in results:
    print(f"Results for C = {result['C']}:")
    print(f"  Weights: {result['Weights']}")
    print(f"  Bias: {result['Bias']}")
    print(f"  Training Error: {result['Train Error']}")
    print(f"  Test Error: {result['Test Error']}")
    print()


# In[18]:


# Question 2b #


import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features 
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Implement SVM
def svm_primal_sgd_v2(X, y, C, epochs, gamma_0):
    weights = np.zeros(X.shape[1])  
    bias = 0  
    n_samples = len(y)

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for t, (xi, target) in enumerate(zip(X, y)):
            learning_rate = gamma_0 / (1 + t)  
            condition = target * (np.dot(weights, xi) + bias)

            if condition < 1:
                
                weights = (1 - learning_rate) * weights + learning_rate * C * target * xi
                bias += learning_rate * C * target
            else:
              
                weights = (1 - learning_rate) * weights

    return weights, bias

# Predict function for SVM
def predict_svm(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)

# Set hyperparameters
epochs = 100  
gamma_0 = 0.1  


C_values = [100/873, 500/873, 700/873]

# Train and test the model for C value
results = []
for C in C_values:
    weights, bias = svm_primal_sgd_v2(X_train, y_train, C, epochs, gamma_0)
    y_pred_train = predict_svm(X_train, weights, bias)
    y_pred_test = predict_svm(X_test, weights, bias)
    
    # Calculate errors
    train_error = np.mean(y_pred_train != y_train)
    test_error = np.mean(y_pred_test != y_test)
    
    results.append({
        'C': C,
        'Weights': weights,
        'Bias': bias,
        'Train Error': train_error,
        'Test Error': test_error
    })

# Print results
for result in results:
    print(f"Results for C = {result['C']}:")
    print(f"  Weights: {result['Weights']}")
    print(f"  Bias: {result['Bias']}")
    print(f"  Training Error: {result['Train Error']}")
    print(f"  Test Error: {result['Test Error']}")
    print()


# In[32]:


# Question 3a #

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Kernel function for linear SVM
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Dual optimization problem for SVM
def svm_dual(X, y, C):
    n_samples = X.shape[0]
    
    # Kernel matrix
    K = np.array([[linear_kernel(xi, xj) for xj in X] for xi in X])
    
    # Define the dual objective function
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)
    
    # Equality constraint: sum(alpha_i * y_i) = 0
    def eq_constraint(alpha):
        return np.dot(alpha, y)
    
    # Bounds for alpha values
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Initial guess for alpha
    alpha0 = np.zeros(n_samples)
    
    # Optimization problem
    result = minimize(
        fun=objective,
        x0=alpha0,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': eq_constraint}
    )
    
    return result.x

# Recover weights and bias from dual solution
def calculate_weights_bias(X, y, alpha, C):
    # Calculate weights (w)
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    
    # Support vectors: alpha_i > 0
    support_vector_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    
    # Calculate bias (b) 
    b = np.mean(y[support_vector_indices] - np.dot(X[support_vector_indices], w))
    
    return w, b

# Hyperparameter values for C
C_values = [100/873, 500/873, 700/873]

# Calculate and print weights and bias for each C
for C in C_values:
    
    alphas = svm_dual(X_train, y_train, C)
    
    # weights and bias
    weights, bias = calculate_weights_bias(X_train, y_train, alphas, C)
    
    # Results
    print(f"Results for C = {C}:")
    print(f"  Weights: {weights}")
    print(f"  Bias: {bias}")
    print()


# In[38]:


# Question 3b #

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load data
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\bank-note\test.csv', header=None)

# Separate features
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.sum((x1 - x2)**2) / gamma)

# Calculate kernel matrix
def calculate_kernel_matrix(X1, X2, gamma):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = gaussian_kernel(X1[i], X2[j], gamma)
    return K

# Optimization 
def svm_dual_kernel(X, y, C, gamma):
    n_samples = X.shape[0]
    
    
    K = calculate_kernel_matrix(X, X, gamma)
    
  
    def objective(alpha):
        return 0.5 * np.sum((alpha * y).reshape(-1, 1) * (alpha * y) * K) - np.sum(alpha)
    
    
    def eq_constraint(alpha):
        return np.dot(alpha, y)
    
    # Bounds for alpha values
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Initial guess for alpha
    alpha0 = np.zeros(n_samples)
    
    
    result = minimize(
        fun=objective,
        x0=alpha0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': eq_constraint}
    )
    
    return result.x

# Prediction function for kernel SVM
def predict_kernel(X_train, X_test, y_train, alpha, gamma, bias):
    y_pred = []
    for x_test in X_test:
        # Calculate kernel values 
        k = np.array([gaussian_kernel(x_test, x_train, gamma) for x_train in X_train])
        # Prediction
        pred = np.sum(alpha * y_train * k) + bias
        y_pred.append(np.sign(pred))
    return np.array(y_pred)

# Calculate bias for kernel SVM
def calculate_bias_kernel(X_train, y_train, alpha, gamma, C):
    
    sv_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    if len(sv_indices) == 0:
        sv_indices = np.where(alpha > 1e-5)[0]
    
    biases = []
    for i in sv_indices:
       
        k = np.array([gaussian_kernel(X_train[i], x_train, gamma) for x_train in X_train])
      
        bias = y_train[i] - np.sum(alpha * y_train * k)
        biases.append(bias)
    
    return np.mean(biases) if biases else 0

# Calculate error rate
def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Hyperparameter values
C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.1, 0.5, 1, 5, 100]

# Store results
results = []

# Test for C and gamma
for C in C_values:
    for gamma in gamma_values:
        print(f"\nTesting C={C:.4f}, gamma={gamma}")
        
        # Train the model
        alphas = svm_dual_kernel(X_train, y_train, C, gamma)
        
        # Calculate bias
        bias = calculate_bias_kernel(X_train, y_train, alphas, gamma, C)
        
        # Make predictions
        y_train_pred = predict_kernel(X_train, X_train, y_train, alphas, gamma, bias)
        y_test_pred = predict_kernel(X_train, X_test, y_train, alphas, gamma, bias)
        
        # Calculate errors
        train_error = calculate_error(y_train, y_train_pred)
        test_error = calculate_error(y_test, y_test_pred)
        
        # Count support vectors
        n_support_vectors = np.sum(alphas > 1e-5)
        
        # Results
        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")
        print(f"Number of support vectors: {n_support_vectors}")
        
        # Save
        results.append({
            'C': C,
            'gamma': gamma,
            'train_error': train_error,
            'test_error': test_error,
            'n_support_vectors': n_support_vectors
        })

# Best result
best_result = min(results, key=lambda x: x['test_error'])
print("\nBest combination based on test error:")
print(f"C = {best_result['C']:.4f}, gamma = {best_result['gamma']}")
print(f"Test error: {best_result['test_error']:.4f}")
print(f"Training error: {best_result['train_error']:.4f}")


# In[40]:


# Question 3d #

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
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.sum((x1 - x2)**2) / gamma)

# Kernel perceptron training
def kernel_perceptron(X, y, gamma, max_iterations=100):
    n_samples = X.shape[0]
    c = np.zeros(n_samples)  
    converged = False
    iterations = 0

    # Precompute kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    
    while not converged and iterations < max_iterations:
        converged = True
        for i in range(n_samples):
            decision = np.sum(c * y * K[:, i])
            if y[i] * decision <= 0: 
                c[i] += 1
                converged = False
        iterations += 1
    
    return c

# Kernel perceptron prediction
def predict_kernel_perceptron(X_train, X_test, y_train, c, gamma):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    predictions = []

    for i in range(n_test):
        decision = 0
        for j in range(n_train):
            decision += c[j] * y_train[j] * gaussian_kernel(X_train[j], X_test[i], gamma)
        predictions.append(np.sign(decision))
    
    return np.array(predictions)

# Calculate error rate
def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Hyperparameter values
gamma_values = [0.1, 0.5, 1, 5, 100]

# Store results
results = []

# Test different values of gamma
for gamma in gamma_values:
    print(f"\nTesting gamma={gamma}")
    
    # Train the Kernel Perceptron
    c = kernel_perceptron(X_train, y_train, gamma)
    
    # Make predictions
    y_train_pred = predict_kernel_perceptron(X_train, X_train, y_train, c, gamma)
    y_test_pred = predict_kernel_perceptron(X_train, X_test, y_train, c, gamma)
    
    # Calculate errors
    train_error = calculate_error(y_train, y_train_pred)
    test_error = calculate_error(y_test, y_test_pred)
    
    # Print results
    print(f"Training error: {train_error:.4f}")
    print(f"Test error: {test_error:.4f}")
    
    # save
    results.append({
        'gamma': gamma,
        'train_error': train_error,
        'test_error': test_error
    })

# Print the best result
best_result = min(results, key=lambda x: x['test_error'])
print("\nBest gamma based on test error:")
print(f"Gamma = {best_result['gamma']}")
print(f"Test error: {best_result['test_error']:.4f}")
print(f"Training error: {best_result['train_error']:.4f}")


# In[ ]:




