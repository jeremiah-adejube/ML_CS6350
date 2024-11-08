#!/usr/bin/env python
# coding: utf-8

# In[41]:


get_ipython().system('jupyter nbconvert --to script Assignment3.ipynb')


# In[15]:


#Question 2a#

import numpy as np
import pandas as pd

# Load the training and test datasets
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\train.csv', header=None)

# Separate features and labels for training and test datasets
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1 if necessary
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Define the standard Perceptron training function with tracking
def train_perceptron(X, y, X_test, y_test, epochs=10):
    weights = np.zeros(X.shape[1])  # Initialize weights to zero
    bias = 0
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            # Perceptron update rule
            update = target * (np.dot(xi, weights) + bias) <= 0
            weights += update * target * xi
            bias += update * target

        # Calculate and display the test error for each epoch
        y_pred = predict(X_test, weights, bias)
        test_error = np.mean(y_pred != y_test)
        print(f"Epoch {epoch + 1}:")
        print(" Learned weight vector:", weights)
        print(" Bias:", bias)
        print(" Average Prediction:", test_error)
        print("-" * 30)
        
    return weights, bias

# Define the prediction function for the Perceptron model
def predict(X, weights, bias):
    return np.where(np.dot(X, weights) + bias >= 0, 1, -1)

# Train the Perceptron model and observe the output per epoch
weights, bias = train_perceptron(X_train, y_train, X_test, y_test)


# In[39]:


# Question 2b #

import numpy as np
import pandas as pd

# Load the training and test datasets
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\test.csv', header=None)

# Separate features and labels for training and test datasets
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1 if necessary
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Define the voted Perceptron training function
def train_voted_perceptron(X, y, epochs=10):
    weights = np.zeros(X.shape[1])  # Initialize weights to zero
    bias = 0
    vote_list = []  # Store each distinct weight vector and count
    count = 1
    
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            if target * (np.dot(xi, weights) + bias) <= 0:
                # Store the current weight vector and its count before updating
                vote_list.append((weights.copy(), bias, count))
                # Update weights and bias
                weights += target * xi
                bias += target
                # Reset count after an update
                count = 1
            else:
                count += 1  # Increment count if no update is made
                
        # Store the final weight and count at the end of each epoch
        vote_list.append((weights.copy(), bias, count))
    
    return vote_list

# Define the voted prediction function
def predict_voted(X, vote_list):
    predictions = np.zeros(X.shape[0])
    for weights, bias, count in vote_list:
        # Compute predictions for the current weights
        predictions += count * np.where(np.dot(X, weights) + bias >= 0, 1, -1)
    return np.where(predictions >= 0, 1, -1)

# Train the Voted Perceptron model
vote_list = train_voted_perceptron(X_train, y_train)

# Make predictions on the test data
y_pred = predict_voted(X_test, vote_list)

# Calculate the test error
test_error = np.mean(y_pred != y_test)

# Output results (to view the detailed break down)
print("Voted Perceptron Results:")
print("List of distinct weight vectors and their counts:")
for i, (weights, bias, count) in enumerate(vote_list):
    print(f"Vector {i + 1}: Weights = {weights}, Bias = {bias}, Count = {count}")

print("\nAverage prediction error on the test dataset:", test_error)


# In[33]:


# Question 2c #

import numpy as np
import pandas as pd

# Load the training and test datasets
train_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\train.csv', header=None)
test_data = pd.read_csv(r'C:\Users\Jeremiah\Desktop\CS Assignment\Homework 3\bank-note\test.csv', header=None)

# Separate features and labels for training and test datasets
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert labels to -1 and 1 if necessary
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Define the average Perceptron training function
def train_average_perceptron(X, y, epochs=10):
    weights = np.zeros(X.shape[1])  # Initialize weights to zero
    bias = 0
    avg_weights = np.zeros(X.shape[1])  # Initialize average weights
    avg_bias = 0

    for epoch in range(epochs):
        for xi, target in zip(X, y):
            # Check if the prediction is incorrect
            if target * (np.dot(xi, weights) + bias) <= 0:
                # Update weights and bias
                weights += target * xi
                bias += target

            # Accumulate the weights and bias to average them
            avg_weights += weights
            avg_bias += bias

    # Return the averaged weights and bias
    avg_weights /= (epochs * len(y))
    avg_bias /= (epochs * len(y))
    return avg_weights, avg_bias

# Define the prediction function for the average Perceptron model
def predict(X, weights, bias):
    return np.where(np.dot(X, weights) + bias >= 0, 1, -1)

# Train the average Perceptron model
avg_weights, avg_bias = train_average_perceptron(X_train, y_train)

# Make predictions on the test data using the averaged weights and bias
y_pred = predict(X_test, avg_weights, avg_bias)

# Calculate the test error
test_error = np.mean(y_pred != y_test)

# Output results
print("Learned averaged weight vector:", avg_weights)
print("Averaged bias:", avg_bias)
print("Average prediction error on the test dataset:", test_error)


# In[ ]:




