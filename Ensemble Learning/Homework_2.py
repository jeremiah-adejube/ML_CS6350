#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Homework_2.ipynb')


# In[1]:


# Question 2a #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# File paths for the training and test data
path_f = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\train.csv'
path_j = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\test.csv'

def preprocess_data(data):
    # Assume the last column is the target variable
    target_col = data.columns[-1]
    
    # Copy the data
    processed_data = data.copy()

    # Identify categorical and numerical columns based on data type
    categorical_cols = processed_data.select_dtypes(include='object').columns
    numerical_cols = processed_data.select_dtypes(exclude='object').columns

    # Replacing unknown values with mode
    for col in categorical_cols:
        mode = processed_data[col].mode()[0]
        processed_data[col] = processed_data[col].replace('unknown', mode)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col != target_col:
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))
    
    # Normalize numerical columns
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
    
    # Encode target variable
    processed_data[target_col] = processed_data[target_col].map({'yes': 1, 'no': -1})
    
    # Rename target column to 'y'
    processed_data = processed_data.rename(columns={target_col: 'y'})
    
    return processed_data

def train_decision_stump(data, weights, attributes):
    best_feature, best_threshold, best_error, best_polarity = None, None, float('inf'), 1
    n_samples = len(data)

    for feature in attributes:
        feature_values = data[feature]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = np.ones(n_samples)
                if polarity == 1:
                    predictions[feature_values < threshold] = -1
                else:
                    predictions[feature_values > threshold] = -1

                weighted_error = np.sum(weights[predictions != data['y']])
                
                if weighted_error < best_error:
                    best_error = weighted_error
                    best_feature = feature
                    best_threshold = threshold
                    best_polarity = polarity

    return best_feature, best_threshold, best_polarity, best_error

def decision_stump_predict(X, feature, threshold, polarity):
    predictions = np.ones(len(X))
    if polarity == 1:
        predictions[X[feature] < threshold] = -1
    else:
        predictions[X[feature] > threshold] = -1
    return predictions

def adaboost(train_data, test_data, attributes, T=500):
    n_samples_train = len(train_data)
    weights = np.full(n_samples_train, 1 / n_samples_train)
    classifiers = []
    alphas = []

    training_errors = []
    test_errors = []
    stump_train_errors = []
    stump_test_errors = []

    for t in range(T):
        feature, threshold, polarity, error = train_decision_stump(train_data, weights, attributes)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        
        train_predictions = decision_stump_predict(train_data, feature, threshold, polarity)
        test_predictions = decision_stump_predict(test_data, feature, threshold, polarity)
        
        # Calculate stump errors
        stump_train_error = np.mean(train_predictions != train_data['y'])
        stump_test_error = np.mean(test_predictions != test_data['y'])
        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)
        
        # Update weights
        weights *= np.exp(-alpha * train_data['y'] * train_predictions)
        weights /= np.sum(weights)

        classifiers.append((feature, threshold, polarity))
        alphas.append(alpha)

        # Aggregate predictions for training and test data
        agg_train_predictions = np.sign(np.sum([alpha * decision_stump_predict(train_data, clf[0], clf[1], clf[2]) 
                                                for alpha, clf in zip(alphas, classifiers)], axis=0))
        agg_test_predictions = np.sign(np.sum([alpha * decision_stump_predict(test_data, clf[0], clf[1], clf[2]) 
                                                for alpha, clf in zip(alphas, classifiers)], axis=0))

        train_error = np.mean(agg_train_predictions != train_data['y'])
        test_error = np.mean(agg_test_predictions != test_data['y'])

        training_errors.append(train_error)
        test_errors.append(test_error)

        if t % 100 == 0:
            print(f"Iteration {t}: Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")

    return classifiers, alphas, training_errors, test_errors, stump_train_errors, stump_test_errors

# Load data
train_data = pd.read_csv(path_f, header=None)
test_data = pd.read_csv(path_j, header=None)

# Preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Get attributes (all columns except 'y')
attributes = [col for col in train_data.columns if col != 'y']

# Train AdaBoost
T = 500
classifiers, alphas, training_errors, test_errors, stump_train_errors, stump_test_errors = adaboost(train_data, test_data, attributes, T=T)

# Plotting AdaBoost errors
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), training_errors, label='Training Error', color='blue', linestyle='--')
plt.plot(range(1, T + 1), test_errors, label='Test Error', color='orange')
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('AdaBoost - Error Rates Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the stump training and test errors for each decision stump
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), stump_train_errors, label='Stump Training Error', color='green')
plt.plot(range(1, T + 1), stump_test_errors, label='Stump Test Error', color='orange')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error Rate')
plt.title('Training and Test Errors of Decision Stumps in Each Iteration')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# Question 2b #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# File paths for the training and test data
path_f = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\train.csv'
path_j = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\test.csv'

# Preprocessing the data by converting numerical features based on a threshold
def preprocessing_bank_data(bank_df, numerical_thresholds, numerical_columns):
    for column in numerical_columns:
        bank_df[column] = bank_df[column].apply(lambda x: 0 if x <= numerical_thresholds[column] else 1)
    return bank_df

# Handling unknown values by replacing them with the mode
def replace_unknown_data(bank_df, categorical_columns_with_unknown_values):
    for column in categorical_columns_with_unknown_values:
        values_with_frequency = Counter(bank_df[column]).most_common(2)
        mode = [value for value, frequency in values_with_frequency if value != 'unknown'][0]
        bank_df[column] = bank_df[column].apply(lambda x: mode if x == 'unknown' else x)
    return bank_df

# Calculate entropy
def cal_entropy(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts(normalize=True)
    return -np.sum(label_counts * np.log2(label_counts))

# Calculate information gain
def cal_infor_gain(data, attribute):
    base_entropy = cal_entropy(data)
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * cal_entropy(subset)
    return base_entropy - weighted_entropy

# Choose the best attribute based on the prediction
def ch_best_attribute(data, attributes):
    gains = {attr: cal_infor_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)

# ID3 Decision Tree algorithm
def id3(data, attributes, max_depth, current_depth=0):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not attributes or (max_depth is not None and current_depth == max_depth):
        return labels.mode()[0]
    
    best_attribute = ch_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            tree[best_attribute][value] = labels.mode()[0]
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            tree[best_attribute][value] = id3(subset, new_attributes, max_depth, current_depth + 1)
    
    return tree

# Prediction function
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0]
    value = instance[attribute]
    
    if value not in tree[attribute]:
        return None
    
    return predict(tree[attribute][value], instance)

# Evaluate the accuracy of the tree
def evaluate(tree, data):
    correct = 0
    for _, row in data.iterrows():
        if predict(tree, row) == row.iloc[-1]:
            correct += 1
    return correct / len(data)

# Bootstrap sampling for bagging
def bootstrap_sample(data):
    n = len(data)
    return data.sample(n=n, replace=True)

# Bagged trees implementation
def bagged_trees(data, attributes, num_trees):
    trees = []
    for _ in range(num_trees):
        sample = bootstrap_sample(data)
        tree = id3(sample, attributes, max_depth=None)
        trees.append(tree)
    return trees

# Predict using bagged trees
def predict_bagged(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    return Counter(predictions).most_common(1)[0][0]

# Evaluate bagged trees
def evaluate_bagged(trees, data):
    correct = 0
    for _, row in data.iterrows():
        if predict_bagged(trees, row) == row.iloc[-1]:
            correct += 1
    return correct / len(data)

# Load and preprocess data
train_data = pd.read_csv(path_f, header=None)
test_data = pd.read_csv(path_j, header=None)

# Assign column names
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = column_names
test_data.columns = column_names

# Define numerical columns and thresholds
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numerical_thresholds = train_data[numerical_columns].median().to_dict()

# Define categorical columns with 'unknown' values
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# Preprocess the data
train_data = preprocessing_bank_data(train_data, numerical_thresholds, numerical_columns)
train_data = replace_unknown_data(train_data, categorical_columns_with_unknown_values)

test_data = preprocessing_bank_data(test_data, numerical_thresholds, numerical_columns)
test_data = replace_unknown_data(test_data, categorical_columns_with_unknown_values)

# Split training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# List of attributes (excluding the label column)
attributes = list(train_data.columns[:-1])

# Initialize lists to store results
num_trees_list = list(range(1, 501, 10)) 
train_errors = []
val_errors = []
test_errors = []

# Train and evaluate bagged trees
for num_trees in num_trees_list:
    trees = bagged_trees(train_data, attributes, num_trees)
    
    train_error = 1 - evaluate_bagged(trees, train_data)
    val_error = 1 - evaluate_bagged(trees, val_data)
    test_error = 1 - evaluate_bagged(trees, test_data)
    
    train_errors.append(train_error)
    val_errors.append(val_error)
    test_errors.append(test_error)
    
    print(f"Number of trees: {num_trees}")
    print(f"  Train error: {train_error:.4f}")
    print(f"  Validation error: {val_error:.4f}")
    print(f"  Test error: {test_error:.4f}")

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(num_trees_list, train_errors, label='Train Error')
plt.plot(num_trees_list, val_errors, label='Validation Error')
plt.plot(num_trees_list, test_errors, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Bagged Trees: Error Rates vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()

# Print final results
print("\nFinal Results:")
print(f"Single Tree Test Error: {1 - evaluate(id3(train_data, attributes, max_depth=None), test_data):.4f}")
print(f"Bagged Trees (500 trees) Test Error: {test_errors[-1]:.4f}")


# In[ ]:


#Question 2c#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# File paths
path_f = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\train.csv'
path_j = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\test.csv'

# Load data
train_data = pd.read_csv(path_f, header=None)
test_data = pd.read_csv(path_j, header=None)

# Assign column names
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = column_names
test_data.columns = column_names

# Convert the target column to binary values: 'yes' to 1 and 'no' to -1
train_data['y'] = train_data['y'].apply(lambda x: 1 if x == 'yes' else -1)
test_data['y'] = test_data['y'].apply(lambda x: 1 if x == 'yes' else -1)

# categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns]

# Separating features and labels
X_train = train_data.drop(columns=['y']).values
y_train = train_data['y'].values
X_test = test_data.drop(columns=['y']).values
y_test = test_data['y'].values

# Number of times to repeat the sampling and bagging procedure
n_repeats = 100
n_samples = 1000
n_trees = 500

# Arrays to store predictions for single trees and bagged trees
single_tree_predictions = np.zeros((len(y_test), n_repeats))
bagged_predictions = np.zeros((len(y_test), n_repeats))

# Run the experiment 100 times
for i in range(n_repeats):
    # STEP 1: Sample 1,000 examples uniformly without replacement from the training dataset
    X_sample, y_sample = resample(X_train, y_train, n_samples=n_samples, replace=False)

    # STEP 2: Train 500 decision trees for bagging
    tree_predictions = np.zeros((len(y_test), n_trees))
    for j in range(n_trees):
        tree = DecisionTreeClassifier()
        tree.fit(X_sample, y_sample)
        tree_predictions[:, j] = tree.predict(X_test)

    # Storing the predictions of the first tree as the single tree prediction
    single_tree_predictions[:, i] = tree_predictions[:, 0]

    # Compute the average prediction of the 500 trees for bagged predictors
    bagged_predictions[:, i] = np.mean(tree_predictions, axis=1)

# Function to compute bias, variance, and squared error
def compute_bias_variance(true_values, predictions):
    avg_predictions = np.mean(predictions, axis=1)
    bias = np.mean((avg_predictions - true_values) ** 2)
    variance = np.mean(np.var(predictions, axis=1))
    squared_error = bias + variance
    return bias, variance, squared_error

# Calculate bias, variance, and squared error for single trees
single_bias, single_variance, single_error = compute_bias_variance(y_test, single_tree_predictions)

# Calculate bias, variance, and squared error for bagged trees
bagged_bias, bagged_variance, bagged_error = compute_bias_variance(y_test, bagged_predictions)

# Print results
print(f"Single Tree Learner - Bias: {single_bias:.4f}, Variance: {single_variance:.4f}, Squared Error: {single_error:.4f}")
print(f"Bagged Trees - Bias: {bagged_bias:.4f}, Variance: {bagged_variance:.4f}, Squared Error: {bagged_error:.4f}")

# Plotting bias, variance, and error for comparison
labels = ['Bias', 'Variance', 'Squared Error']
single_tree_values = [single_bias, single_variance, single_error]
bagged_tree_values = [bagged_bias, bagged_variance, bagged_error]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, single_tree_values, width, label='Single Tree')
plt.bar(x + width/2, bagged_tree_values, width, label='Bagged Trees')
plt.xlabel('Error Components')
plt.ylabel('Values')
plt.title('Bias-Variance Decomposition: Single Tree vs Bagged Trees')
plt.xticks(x, labels)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


#Question 2d#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# File paths for the training and test data
path_train = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\train.csv'
path_test = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\test.csv'

# Preprocessing the data by converting numerical features based on a threshold
def preprocessing_bank_data(bank_df, numerical_thresholds, numerical_columns):
    for column in numerical_columns:
        bank_df[column] = bank_df[column].apply(lambda x: 0 if x <= numerical_thresholds[column] else 1)
    return bank_df

# Handling unknown values by replacing them with the mode
def replace_unknown_data(bank_df, categorical_columns_with_unknown_values):
    for column in categorical_columns_with_unknown_values:
        values_with_frequency = Counter(bank_df[column]).most_common(2)
        mode = [value for value, frequency in values_with_frequency if value != 'unknown'][0]
        bank_df[column] = bank_df[column].apply(lambda x: mode if x == 'unknown' else x)
    return bank_df

# Calculate entropy
def cal_entropy(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts(normalize=True)
    return -np.sum(label_counts * np.log2(label_counts + 1e-9))

# Calculate information gain
def cal_infor_gain(data, attribute):
    base_entropy = cal_entropy(data)
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * cal_entropy(subset)
    return base_entropy - weighted_entropy

# Choose the best attribute based on information gain
def ch_best_attribute(data, attributes):
    gains = {attr: cal_infor_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)

# ID3 Decision Tree algorithm for Random Forest
def id3_random_forest(data, all_attributes, feature_subset_size, max_depth, current_depth=0):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not all_attributes or current_depth == max_depth:
        return labels.mode()[0]
    
    # Randomly select a subset of features
    attributes = random.sample(all_attributes, min(feature_subset_size, len(all_attributes)))
    
    best_attribute = ch_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            tree[best_attribute][value] = labels.mode()[0]
        else:
            tree[best_attribute][value] = id3_random_forest(subset, all_attributes, feature_subset_size, max_depth, current_depth + 1)
    
    return tree

# Prediction function for a single tree
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0]
    value = instance[attribute]

    if value in tree[attribute]:
        return predict(tree[attribute][value], instance)
    else:
        # Handle the case when the value is not in the tree
        values = [v for v in tree[attribute].values() if not isinstance(v, dict)]
        if values:
            return Counter(values).most_common(1)[0][0]
        else:
            # If there are no non-dict values, recursively check deeper
            return predict(random.choice(list(tree[attribute].values())), instance)

# Random Forest creation
def random_forest(data, all_attributes, n_trees, feature_subset_size, max_depth):
    forest = []
    for _ in range(n_trees):
        # Bootstrap sampling
        bootstrap_sample = data.sample(n=len(data), replace=True)
        tree = id3_random_forest(bootstrap_sample, all_attributes, feature_subset_size, max_depth)
        forest.append(tree)
    return forest

# Prediction function for Random Forest
def predict_forest(forest, instance):
    predictions = [predict(tree, instance) for tree in forest]
    return max(set(predictions), key=predictions.count)

# Evaluate Random Forest
def evaluate_forest(forest, data):
    correct = 0
    for _, row in data.iterrows():
        if predict_forest(forest, row) == row.iloc[-1]:
            correct += 1
    return correct / len(data)

# Main experiment function
def run_experiment(train_data, test_data, attributes, n_trees_range, feature_subset_sizes, max_depth):
    results = []

    for feature_subset_size in feature_subset_sizes:
        train_errors = []
        test_errors = []
        for n_trees in n_trees_range:
            forest = random_forest(train_data, attributes, n_trees, feature_subset_size, max_depth)
            train_error = 1 - evaluate_forest(forest, train_data)
            test_error = 1 - evaluate_forest(forest, test_data)
            train_errors.append(train_error)
            test_errors.append(test_error)
            print(f"Feature subset size: {feature_subset_size}, Trees: {n_trees}, Train error: {train_error:.4f}, Test error: {test_error:.4f}")
        results.append((feature_subset_size, train_errors, test_errors))
    
    return results

# Load and preprocess data
train_data = pd.read_csv(path_train, header=None)
test_data = pd.read_csv(path_test, header=None)

# Assign column names 
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = column_names
test_data.columns = column_names

# Define numerical columns and thresholds
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numerical_thresholds = train_data[numerical_columns].median().to_dict()

# Define categorical columns with 'unknown' values
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# Preprocess the data
train_data = preprocessing_bank_data(train_data, numerical_thresholds, numerical_columns)
train_data = replace_unknown_data(train_data, categorical_columns_with_unknown_values)

test_data = preprocessing_bank_data(test_data, numerical_thresholds, numerical_columns)
test_data = replace_unknown_data(test_data, categorical_columns_with_unknown_values)

# List of attributes (excluding the label column)
attributes = list(train_data.columns[:-1])

# Experiment parameters
n_trees_range = range(1, 501, 10)  # From 1 to 500 trees, step 10
feature_subset_sizes = [2, 4, 6]
max_depth = 10  # You can adjust this

# Run the experiment
results = run_experiment(train_data, test_data, attributes, n_trees_range, feature_subset_sizes, max_depth)

# Plotting Random Forest results
plt.figure(figsize=(12, 8))
for feature_subset_size, train_errors, test_errors in results:
    plt.plot(n_trees_range, train_errors, label=f'Train (Features: {feature_subset_size})')
    plt.plot(n_trees_range, test_errors, label=f'Test (Features: {feature_subset_size})')

plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Random Forest Performance')
plt.legend()
plt.grid(True)
plt.savefig('random_forest_performance.png')
plt.close()

# Comparison with bagged trees (feature_subset_size = len(attributes))
bagged_results = run_experiment(train_data, test_data, attributes, n_trees_range, [len(attributes)], max_depth)

# Plotting Random Forest vs Bagged Trees
plt.figure(figsize=(12, 8))
_, bagged_train_errors, bagged_test_errors = bagged_results[0]
plt.plot(n_trees_range, bagged_train_errors, label='Bagged Trees (Train)')
plt.plot(n_trees_range, bagged_test_errors, label='Bagged Trees (Test)')
for feature_subset_size, _, test_errors in results:
    plt.plot(n_trees_range, test_errors, label=f'Random Forest (Features: {feature_subset_size})')

plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Random Forest vs Bagged Trees Performance')
plt.legend()
plt.grid(True)
plt.close()


# In[ ]:


# Question 2e #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.preprocessing import LabelEncoder

# Preprocessing the data by converting numerical features based on a threshold
def preprocessing_bank_data(bank_df, numerical_thresholds, numerical_columns):
    for column in numerical_columns:
        bank_df[column] = bank_df[column].apply(lambda x: 0 if x <= numerical_thresholds[column] else 1)
    return bank_df

# Handling unknown values by replacing them with the mode
def replace_unknown_data(bank_df, categorical_columns_with_unknown_values):
    for column in categorical_columns_with_unknown_values:
        values_with_frequency = Counter(bank_df[column]).most_common(2)
        mode = [value for value, frequency in values_with_frequency if value != 'unknown'][0]
        bank_df[column] = bank_df[column].apply(lambda x: mode if x == 'unknown' else x)
    return bank_df

# Calculate entropy
def cal_entropy(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts(normalize=True)
    return -np.sum(label_counts * np.log2(label_counts + 1e-9))

# Calculate information gain
def cal_infor_gain(data, attribute):
    base_entropy = cal_entropy(data)
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * cal_entropy(subset)
    return base_entropy - weighted_entropy

# Choose the best attribute based on information gain
def ch_best_attribute(data, attributes):
    gains = {attr: cal_infor_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)

# ID3 Decision Tree algorithm for Random Forest
def id3_random_forest(data, all_attributes, feature_subset_size, max_depth, current_depth=0):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not all_attributes or current_depth == max_depth:
        return labels.mode()[0]
    
    attributes = random.sample(all_attributes, min(feature_subset_size, len(all_attributes)))
    best_attribute = ch_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            tree[best_attribute][value] = labels.mode()[0]
        else:
            tree[best_attribute][value] = id3_random_forest(subset, all_attributes, feature_subset_size, max_depth, current_depth + 1)
    
    return tree

# Prediction function for a single tree
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0]
    value = instance[attribute]

    if value in tree[attribute]:
        return predict(tree[attribute][value], instance)
    else:
        values = [v for v in tree[attribute].values() if not isinstance(v, dict)]
        if values:
            return Counter(values).most_common(1)[0][0]
        else:
            return predict(random.choice(list(tree[attribute].values())), instance)

# Random Forest creation
def random_forest(data, all_attributes, n_trees, feature_subset_size, max_depth):
    forest = []
    for _ in range(n_trees):
        bootstrap_sample = data.sample(n=len(data), replace=True)
        tree = id3_random_forest(bootstrap_sample, all_attributes, feature_subset_size, max_depth)
        forest.append(tree)
    return forest

# Updated Prediction function for Random Forest
def predict_forest(forest, instance):
    predictions = [predict(tree, instance) for tree in forest]
    
    # Try to convert predictions to numeric, replacing non-numeric with NaN
    numeric_predictions = pd.to_numeric(predictions, errors='coerce')
    
    # Remove NaN values
    valid_predictions = numeric_predictions[~np.isnan(numeric_predictions)]
    
    # If no valid numeric predictions, return NaN
    if len(valid_predictions) == 0:
        return np.nan
    
    return np.argmax(np.bincount(valid_predictions.astype(int)))

# Evaluate Random Forest
def evaluate_forest(forest, data):
    correct = 0
    for _, row in data.iterrows():
        if predict_forest(forest, row) == row.iloc[-1]:
            correct += 1
    return correct / len(data)

# Estimate bias and variance for the forest
def estimate_bias_variance_forest(train_data, test_data, attributes, n_trees, feature_subset_size, max_depth, n_samples=100):
    predictions = []
    
    for _ in range(n_samples):
        forest = random_forest(train_data, attributes, n_trees, feature_subset_size, max_depth)
        preds = test_data.apply(lambda row: predict_forest(forest, row), axis=1)
        
        # Convert predictions to numeric values
        preds = pd.to_numeric(preds, errors='coerce')
        predictions.append(preds)
    
    predictions = np.array(predictions)
    
    expected_predictions = np.mean(predictions, axis=0)
    true_labels = pd.to_numeric(test_data.iloc[:, -1], errors='coerce').values
    
    # Ensure both arrays are of the same shape
    mask = np.isnan(expected_predictions) | np.isnan(true_labels)
    expected_predictions = np.ma.array(expected_predictions, mask=mask)
    true_labels = np.ma.array(true_labels, mask=mask)
    
    # Calculate bias, variance, and squared error
    bias = np.mean((expected_predictions - true_labels) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    squared_error = bias + variance
    
    return bias, variance, squared_error

# Run bias and variance experiment for both single trees and forests
def run_bias_variance_experiment(train_data, test_data, attributes, n_trees_range, feature_subset_sizes, max_depth):
    results = []

    for feature_subset_size in feature_subset_sizes:
        for n_trees in n_trees_range:
            bias, variance, squared_error = estimate_bias_variance_forest(train_data, test_data, attributes, n_trees, feature_subset_size, max_depth)
            results.append((feature_subset_size, n_trees, bias, variance, squared_error))
    
    return results

# Load and preprocess data
path_train = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\train.csv'
path_test = r'C:\Users\Jeremiah\Desktop\CS Assignment\bank\test.csv'
train_data = pd.read_csv(path_train, header=None)
test_data = pd.read_csv(path_test, header=None)

# Assign column names
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = column_names
test_data.columns = column_names

# Convert label column 'y' to numeric using LabelEncoder
label_encoder = LabelEncoder()
train_data['y'] = label_encoder.fit_transform(train_data['y'])
test_data['y'] = label_encoder.transform(test_data['y'])

# Define numerical columns and thresholds
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numerical_thresholds = train_data[numerical_columns].median().to_dict()
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# Preprocess data
train_data = preprocessing_bank_data(train_data, numerical_thresholds, numerical_columns)
train_data = replace_unknown_data(train_data, categorical_columns_with_unknown_values)

test_data = preprocessing_bank_data(test_data, numerical_thresholds, numerical_columns)
test_data = replace_unknown_data(test_data, categorical_columns_with_unknown_values)

# Attributes for the experiment
attributes = list(train_data.columns[:-1])

# Experiment parameters
n_trees_range = range(1, 501, 10)
feature_subset_sizes = [2, 4, 6]
max_depth = 10

# Run bias-variance experiment
bias_variance_results = run_bias_variance_experiment(train_data, test_data, attributes, n_trees_range, feature_subset_sizes, max_depth)

# Plotting the results
plt.figure(figsize=(15, 5))

for i, feature_subset_size in enumerate(feature_subset_sizes):
    subset_results = [r for r in bias_variance_results if r[0] == feature_subset_size]
    n_trees = [r[1] for r in subset_results]
    bias = [r[2] for r in subset_results]
    variance = [r[3] for r in subset_results]
    squared_error = [r[4] for r in subset_results]

    plt.subplot(1, 3, i+1)
    plt.plot(n_trees, bias, label='Bias', color='blue')
    plt.plot(n_trees, variance, label='Variance', color='red')
    plt.plot(n_trees, squared_error, label='Squared Error', color='green')
    
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title(f'Bias-Variance Tradeoff (Feature Subset Size: {feature_subset_size})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png')
plt.show()

# Print the optimal number of trees for each feature subset size
for feature_subset_size in feature_subset_sizes:
    subset_results = [r for r in bias_variance_results if r[0] == feature_subset_size]
    optimal_result = min(subset_results, key=lambda x: x[4])  # Minimum squared error
    print(f"Optimal number of trees for feature subset size {feature_subset_size}: {optimal_result[1]}")
    print(f"Minimum squared error: {optimal_result[4]:.4f}")
    print(f"Bias: {optimal_result[2]:.4f}, Variance: {optimal_result[3]:.4f}")
    print()

# Calculate and print the overall best configuration
best_result = min(bias_variance_results, key=lambda x: x[4])
print("Overall best configuration:")
print(f"Feature subset size: {best_result[0]}")
print(f"Number of trees: {best_result[1]}")
print(f"Minimum squared error: {best_result[4]:.4f}")
print(f"Bias: {best_result[2]:.4f}, Variance: {best_result[3]:.4f}")


# In[ ]:


# Question 3 #

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
default_of_credit_card_client = r'C:\Users\Jeremiah\Desktop\CS Assignment\default of credit card clients.csv'
df = pd.read_csv(default_of_credit_card_client)

# Remove the 'ID' column if it exists
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Ensure all columns are numeric and drop any rows with NaN values
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Split dataset into features and target variable
X = df.drop(columns=['Y'])  # Assuming 'Y' is the target column
y = df['Y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6000, train_size=24000, random_state=42)


# Initialize models with n_estimators set to 500
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, random_state=42)
random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500, random_state=42)

# Optimized Error Plotting Function for Models without staged_predict
def plot_model_errors_manual(model, X_train, y_train, X_test, y_test, model_name):
    train_errors = []
    test_errors = []
    
    for i in range(1, model.n_estimators + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_errors.append(1 - accuracy_score(y_train, y_pred_train))
        test_errors.append(1 - accuracy_score(y_test, y_pred_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, model.n_estimators + 1), train_errors, label='Training Error', linestyle='--')
    plt.plot(range(1, model.n_estimators + 1), test_errors, label='Test Error', linestyle='-')
    plt.title(f'{model_name} - Error Rates Over Iterations')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_error_plot.png')
    plt.close()

# Optimized Error Plotting Function for AdaBoost using staged_predict
def plot_model_errors_adaboost(model, X_train, y_train, X_test, y_test, model_name):
    train_errors = []
    test_errors = []
    
    model.fit(X_train, y_train)
    for y_pred_train, y_pred_test in zip(model.staged_predict(X_train), model.staged_predict(X_test)):
        train_errors.append(1 - accuracy_score(y_train, y_pred_train))
        test_errors.append(1 - accuracy_score(y_test, y_pred_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, model.n_estimators + 1), train_errors, label='Training Error', linestyle='--')
    plt.plot(range(1, model.n_estimators + 1), test_errors, label='Test Error', linestyle='-')
    plt.title(f'{model_name} - Error Rates Over Iterations')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_error_plot.png')
    plt.close()

# Train a single decision tree for comparison
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_tree_train_error = 1 - accuracy_score(y_train, single_tree.predict(X_train))
single_tree_test_error = 1 - accuracy_score(y_test, single_tree.predict(X_test))

# Plotting errors for Bagging and Random Forest (no staged_predict, manual loop)
plot_model_errors_manual(bagging, X_train, y_train, X_test, y_test, "Bagged Trees")
plot_model_errors_manual(random_forest, X_train, y_train, X_test, y_test, "Random Forest")

# Plotting errors for AdaBoost (uses staged_predict)
plot_model_errors_adaboost(adaboost, X_train, y_train, X_test, y_test, "AdaBoost")

# Final Comparison Plot
plt.figure(figsize=(12, 8))

# Plot ensemble method errors
for model, name in zip([bagging, random_forest, adaboost], ['Bagged Trees', 'Random Forest', 'AdaBoost']):
    if name == "AdaBoost":
        model.fit(X_train, y_train)
        test_errors = [1 - accuracy_score(y_test, y_pred) for y_pred in model.staged_predict(X_test)]
    else:
        test_errors = []
        for i in range(1, model.n_estimators + 1):
            model.set_params(n_estimators=i)
            model.fit(X_train, y_train)
            test_errors.append(1 - accuracy_score(y_test, model.predict(X_test)))
    
    plt.plot(range(1, model.n_estimators + 1), test_errors, label=f'{name} (Test)')

# Plot single decision tree error
plt.axhline(y=single_tree_test_error, color='r', linestyle='--', label='Single Decision Tree (Test)')

plt.title('Comparison of Ensemble Methods and Single Decision Tree')
plt.xlabel('Number of Estimators')
plt.ylabel('Test Error Rate')
plt.legend()
plt.grid(True)
plt.close()


# In[ ]:


#Question 4#

import numpy as np
import matplotlib.pyplot as plt

# Loading the data
train_data = np.loadtxt(r'C:\Users\Jeremiah\Desktop\CS Assignment\cement\train.csv', delimiter=',')
test_data = np.loadtxt(r'C:\Users\Jeremiah\Desktop\CS Assignment\cement\test.csv', delimiter=',')

# Spliting the data into features
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Adding the bias term to the feature matrices
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Cost function for linear regression
def cost_function(X, y, w):
    return np.mean((X.dot(w) - y) ** 2) / 2

# Batch Gradient Descent implementation
def batch_gradient_descent(X, y, learning_rate, tolerance=1e-6, max_iterations=10000):
    w = np.zeros(X.shape[1])
    costs = []
    
    for i in range(max_iterations):
        gradient = X.T.dot(X.dot(w) - y) / X.shape[0]
        w_new = w - learning_rate * gradient
        
        if np.linalg.norm(w_new - w) < tolerance:
            break
        
        w = w_new
        costs.append(cost_function(X, y, w))
    
    return w, costs

# Stochastic Gradient Descent (SGD) implementation
def stochastic_gradient_descent(X, y, learning_rate, max_iterations=10000):
    w = np.zeros(X.shape[1])
    costs = []
    
    for i in range(max_iterations):
        idx = np.random.randint(0, X.shape[0])
        xi, yi = X[idx:idx+1], y[idx]
        gradient = xi.T.dot(xi.dot(w) - yi)
        w = w - learning_rate * gradient
        
        if i % 100 == 0:  # Calculate cost every 100 iterations to reduce computation
            costs.append(cost_function(X, y, w))
    
    return w, costs

# Analytical solution for linear regression
def analytical_solution(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Batch Gradient Descent
learning_rate_bgd = 0.001  # Adjust this as needed
w_bgd, costs_bgd = batch_gradient_descent(X_train, y_train, learning_rate_bgd)

# Stochastic Gradient Descent
learning_rate_sgd = 0.001  # Adjust this as needed
w_sgd, costs_sgd = stochastic_gradient_descent(X_train, y_train, learning_rate_sgd)

# Analytical Solution
w_analytical = analytical_solution(X_train, y_train)

# Plot cost function for Batch Gradient Descent
plt.figure(figsize=(10, 5))
plt.plot(costs_bgd)
plt.title('Cost Function vs. Iterations (Batch Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Plot cost function for Stochastic Gradient Descent
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(costs_sgd) * 100, 100), costs_sgd)
plt.title('Cost Function vs. Iterations (Stochastic Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Print results for all methods
print("Batch Gradient Descent:")
print("Weight vector:", w_bgd)
print("Test cost:", cost_function(X_test, y_test, w_bgd))

print("\nStochastic Gradient Descent:")
print("Weight vector:", w_sgd)
print("Test cost:", cost_function(X_test, y_test, w_sgd))

print("\nAnalytical Solution:")
print("Weight vector:", w_analytical)
print("Test cost:", cost_function(X_test, y_test, w_analytical))

