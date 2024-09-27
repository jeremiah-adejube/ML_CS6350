#!/usr/bin/env python
# coding: utf-8

# In[77]:


get_ipython().system('jupyter nbconvert --to script Assignment1A.ipynb')


# In[42]:


#******* Question 1*******#

import pandas as pd
import numpy as np
from math import log2

# Hardcoding the data cuz I'm too lazy to read from a file
data = pd.DataFrame([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0]
], columns=['x1', 'x2', 'x3', 'x4', 'y'])

# Calculate entropy - pretty standard stuff
def entropy(y):
    if len(y) == 0:
        return 0
    p = y.value_counts() / len(y)
    return -sum(p * np.log2(p))

# Information gain calculation
def info_gain(X, y, feature):
    total_entropy = entropy(y)
    
    weighted_entropy = 0
    for value in X[feature].unique():
        subset_y = y[X[feature] == value]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)
    
    return total_entropy - weighted_entropy

# The main ID3 algorithm - recursion is fun!
def id3(X, y, features, depth=0):
    indent = "  " * depth  # For pretty printing
    print(f"{indent}Entropy: {entropy(y):.4f}")
    
    # Base cases
    if len(y.unique()) == 1 or len(features) == 0:
        return y.mode().iloc[0]
    
    # Calculate info gain for each feature
    gains = {}
    for feature in features:
        gain = info_gain(X, y, feature)
        gains[feature] = gain
        print(f"{indent}Gain for {feature}: {gain:.4f}")
    
    # Pick the best feature
    best_feature = max(gains, key=gains.get)
    print(f"{indent}Splitting on: {best_feature}")
    
    # Build the tree recursively
    tree = {best_feature: {}}
    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        subset_y = y[X[best_feature] == value]
        new_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(subset_X, subset_y, new_features, depth + 1)
    
    return tree

# Convert tree to boolean expression (this part's a bit tricky)
def tree_to_bool(tree):
    def traverse(node, path):
        if isinstance(node, dict):
            feature, branches = next(iter(node.items()))
            exprs = []
            for value, subtree in branches.items():
                new_path = path + [f"{feature}={value}"]
                exprs.extend(traverse(subtree, new_path))
            return exprs
        else:
            return [' AND '.join(path)] if node == 1 else []

    exprs = traverse(tree, [])
    return ' OR '.join(f"({expr})" for expr in exprs)

# Let's run this thing!
print("Building the decision tree:")
print("--------------------------")
X = data.drop('y', axis=1)
y = data['y']
features = X.columns.tolist()
decision_tree = id3(X, y, features)

print("\nBoolean expression (hope this is right):")
print(tree_to_bool(decision_tree))


# In[26]:


import pandas as pd
from collections import Counter
import io

# Data as a string (CSV format)
data_str = """Outlook,Temperature,Humidity,Wind,Play?
S,H,H,W,-
S,H,H,S,-
O,H,H,W,+
R,M,H,W,+
R,C,N,W,+
R,C,N,S,-
O,C,N,S,+
S,M,H,W,-
S,C,N,W,+
R,M,N,W,+
S,M,N,S,+
O,M,H,S,+
O,H,N,W,+
R,M,H,S,-"""

# Convert string to DataFrame
data = pd.read_csv(io.StringIO(data_str))

def calculate_me(labels):
    count = Counter(labels)
    total = len(labels)
    return min(count['+'] / total, count['-'] / total)

def calculate_attribute_me(data, attribute):
    weighted_me = 0
    total_samples = len(data)
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        subset_size = len(subset)
        subset_me = calculate_me(subset['Play?'])
        weighted_me += (subset_size / total_samples) * subset_me
    return weighted_me

def calculate_gain(data, attribute):
    root_me = calculate_me(data['Play?'])
    attr_me = calculate_attribute_me(data, attribute)
    return root_me - attr_me

def print_me_and_gain(data, attributes):
    root_me = calculate_me(data['Play?'])
    print(f"ME(root) = {root_me:.3f}")
    for attr in attributes:
        attr_me = calculate_attribute_me(data, attr)
        gain = root_me - attr_me
        print(f"{attr} (ME) = {attr_me:.3f} (IG) = {gain:.3f}")

def build_tree(data):
    print("First Stage:")
    print_me_and_gain(data, ['Outlook', 'Temperature', 'Humidity', 'Wind'])
    
    print("\nSplit on outlook")
    tree = {'Outlook': {}}
    
    for outlook in data['Outlook'].unique():
        outlook_data = data[data['Outlook'] == outlook]
        if outlook == 'O':
            tree['Outlook'][outlook] = '+'
            print(f"{outlook} = Yes (Leafnote)")
        else:
            print(f"{outlook} = Further Splitting")
            tree['Outlook'][outlook] = {}
    
    print("\nSecond stage (Rainy branch):")
    rainy_data = data[data['Outlook'] == 'R']
    print_me_and_gain(rainy_data, ['Temperature', 'Humidity', 'Wind'])
    
    print("\nSecond stage (Sunny branch):")
    sunny_data = data[data['Outlook'] == 'S']
    print_me_and_gain(sunny_data, ['Temperature', 'Humidity', 'Wind'])
    
    print("\nUnder Rainy Split on Wind")
    tree['Outlook']['R']['Wind'] = {}
    for wind in rainy_data['Wind'].unique():
        wind_data = rainy_data[rainy_data['Wind'] == wind]
        label = '+' if wind == 'W' else '-'
        tree['Outlook']['R']['Wind'][wind] = label
        print(f"{wind} = {'Yes' if label == '+' else 'No'} (Leafnote)")
    
    print("\nUnder Sunny Split on Humidity")
    tree['Outlook']['S']['Humidity'] = {}
    for humidity in sunny_data['Humidity'].unique():
        humidity_data = sunny_data[sunny_data['Humidity'] == humidity]
        label = '+' if humidity == 'N' else '-'
        tree['Outlook']['S']['Humidity'][humidity] = label
        print(f"{humidity} = {'Yes' if label == '+' else 'No'} (Leafnote)")
    
    return tree

print("Question 2a:\n")
decision_tree = build_tree(data)


# In[28]:


import pandas as pd
from collections import Counter
import io

# Data as a string (CSV format)
data_str = """Outlook,Temperature,Humidity,Wind,Play?
S,H,H,W,-
S,H,H,S,-
O,H,H,W,+
R,M,H,W,+
R,C,N,W,+
R,C,N,S,-
O,C,N,S,+
S,M,H,W,-
S,C,N,W,+
R,M,N,W,+
S,M,N,S,+
O,M,H,S,+
O,H,N,W,+
R,M,H,S,-"""

# Convert string to DataFrame
data = pd.read_csv(io.StringIO(data_str))

def calculate_gini(labels):
    count = Counter(labels)
    total = len(labels)
    return 1 - sum((count[label] / total) ** 2 for label in count)

def calculate_attribute_gini(data, attribute):
    weighted_gini = 0
    total_samples = len(data)
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        subset_size = len(subset)
        subset_gini = calculate_gini(subset['Play?'])
        weighted_gini += (subset_size / total_samples) * subset_gini
    return weighted_gini

def calculate_gain(data, attribute):
    root_gini = calculate_gini(data['Play?'])
    attr_gini = calculate_attribute_gini(data, attribute)
    return root_gini - attr_gini

def print_gini_and_gain(data, attributes):
    root_gini = calculate_gini(data['Play?'])
    print(f"Gini(root) = {root_gini:.3f}")
    for attr in attributes:
        attr_gini = calculate_attribute_gini(data, attr)
        gain = root_gini - attr_gini
        print(f"{attr} (Gini) = {attr_gini:.3f} (IG) = {gain:.3f}")

def build_tree(data):
    print("First Stage:")
    print_gini_and_gain(data, ['Outlook', 'Temperature', 'Humidity', 'Wind'])
    
    print("\nSplit on outlook")
    tree = {'Outlook': {}}
    
    for outlook in data['Outlook'].unique():
        outlook_data = data[data['Outlook'] == outlook]
        if outlook == 'O':
            tree['Outlook'][outlook] = '+'
            print(f"{outlook} = Yes (Leafnote)")
        else:
            print(f"{outlook} = Further Splitting")
            tree['Outlook'][outlook] = {}
    
    print("\nSecond stage (Rainy branch):")
    rainy_data = data[data['Outlook'] == 'R']
    print_gini_and_gain(rainy_data, ['Temperature', 'Humidity', 'Wind'])
    
    print("\nSecond stage (Sunny branch):")
    sunny_data = data[data['Outlook'] == 'S']
    print_gini_and_gain(sunny_data, ['Temperature', 'Humidity', 'Wind'])
    
    print("\nUnder Rainy Split on Wind")
    tree['Outlook']['R']['Wind'] = {}
    for wind in rainy_data['Wind'].unique():
        wind_data = rainy_data[rainy_data['Wind'] == wind]
        label = '+' if wind == 'W' else '-'
        tree['Outlook']['R']['Wind'][wind] = label
        print(f"{wind} = {'Yes' if label == '+' else 'No'} (Leafnote)")
    
    print("\nUnder Sunny Split on Humidity")
    tree['Outlook']['S']['Humidity'] = {}
    for humidity in sunny_data['Humidity'].unique():
        humidity_data = sunny_data[sunny_data['Humidity'] == humidity]
        label = '+' if humidity == 'N' else '-'
        tree['Outlook']['S']['Humidity'][humidity] = label
        print(f"{humidity} = {'Yes' if label == '+' else 'No'} (Leafnote)")
    
    return tree

print("Question 2b:\n")
decision_tree = build_tree(data)


# In[48]:


#**********Question 3(a)**********#

# I used the most common attribute in this dataset "rainy" because comparing with other attributes and following the trends rainy fits best 
# and has the most likelyhood of occuring. 

#**********Question 3(b)**********#

# The most common value among the training instance with the same label their play attribute is yes is "rainy" because because it has 3 yes and 2 no as
# compared to sunny with 2 yes and 3 no. 

import pandas as pd
import math
from collections import Counter
import io

def entropy(labels):
    """Calculate entropy of the given labels."""
    label_counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in label_counts.values())

def attribute_entropy(data, attr):
    """Calculate weighted entropy for a given attribute."""
    total = len(data)
    return sum(
        len(subset) / total * entropy(subset['Play?'])
        for _, subset in data.groupby(attr)
    )

def info_gain(data, attr):
    """Calculate information gain for an attribute."""
    return entropy(data['Play?']) - attribute_entropy(data, attr)

def print_entropy_and_gain(data, attrs):
    """Print entropy and information gain for given attributes."""
    root_entropy = entropy(data['Play?'])
    print(f"Root entropy: {root_entropy:.3f}")
    
    for attr in attrs:
        attr_entropy = attribute_entropy(data, attr)
        gain = root_entropy - attr_entropy
        print(f"{attr:12} Entropy: {attr_entropy:.3f}  Info Gain: {gain:.3f}")

def build_decision_tree(data):
    """Build and print the decision tree."""
    print("First Stage:")
    print_entropy_and_gain(data, ['Outlook', 'Temperature', 'Humidity', 'Wind'])
    
    tree = {'Outlook': {}}
    print("\nSplitting on Outlook")
    
    for outlook in data['Outlook'].unique():
        outlook_data = data[data['Outlook'] == outlook]
        if outlook == 'O':
            tree['Outlook'][outlook] = '+'
            print(f"  {outlook}: Yes (Leaf node)")
        else:
            print(f"  {outlook}: Needs further splitting")
            tree['Outlook'][outlook] = {}
    
    print("\nSecond stage - Rainy branch:")
    rainy_data = data[data['Outlook'] == 'R']
    print_entropy_and_gain(rainy_data, ['Temperature', 'Humidity', 'Wind'])
    
    print("\nSecond stage - Sunny branch:")
    sunny_data = data[data['Outlook'] == 'S']
    print_entropy_and_gain(sunny_data, ['Temperature', 'Humidity', 'Wind'])
    
    # Handle Rainy branch
    print("\nRainy branch - Splitting on Wind")
    tree['Outlook']['R'] = {'Wind': {}}
    for wind in rainy_data['Wind'].unique():
        decision = '+' if wind == 'W' else '-'
        tree['Outlook']['R']['Wind'][wind] = decision
        print(f"  Wind {wind}: {'Yes' if decision == '+' else 'No'} (Leaf node)")
    
    # Handle Sunny branch
    print("\nSunny branch - Splitting on Humidity")
    tree['Outlook']['S'] = {'Humidity': {}}
    for humidity in sunny_data['Humidity'].unique():
        decision = '+' if humidity == 'N' else '-'
        tree['Outlook']['S']['Humidity'][humidity] = decision
        print(f"  Humidity {humidity}: {'Yes' if decision == '+' else 'No'} (Leaf node)")
    
    return tree

# Data preparation
raw_data = ''',Outlook,Temperature,Humidity,Wind,Play?
1,S,H,H,W,-
2,S,H,H,S,-
3,O,H,H,W,+
4,R,M,H,W,+
5,R,C,N,W,+
6,R,C,N,S,-
7,O,C,N,S,+
8,S,M,H,W,-
9,S,C,N,W,+
10,R,M,N,W,+
11,S,M,N,S,+
12,O,M,H,S,+
13,O,H,N,W,+
14,R,M,H,S,-
15,R,M,N,W,+'''

# Main execution
if __name__ == "__main__":
    print("Question 3a \n")
    df = pd.read_csv(io.StringIO(raw_data))
    tree = build_decision_tree(df)


# In[20]:


#**********Question 3(c)**********#

import pandas as pd
import numpy as np
from collections import Counter

def entropy(probabilities):
    return -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

def cal_feaE_and_IG(data, feature):
    feature_counts = Counter(data[feature].dropna())
    total_known = sum(feature_counts.values())
    
    fractional_counts = {k: v / total_known for k, v in feature_counts.items()} if data[feature].isnull().any() else {}
    
    new_counts = {k: v + fractional_counts.get(k, 0) for k, v in feature_counts.items()}
    
    total_instances = len(data)
    P_count = sum(data['Play?'] == '+')
    N_count = total_instances - P_count
    total_entropy = entropy([P_count/total_instances, N_count/total_instances])
    
    weighted_entropy = 0
    for value, count in new_counts.items():
        subset = data[data[feature] == value]
        subset_positive = sum(subset['Play?'] == '+')
        subset_negative = len(subset) - subset_positive
        
        if value in fractional_counts:
            subset_positive += fractional_counts[value] * (P_count / total_instances)
            subset_negative += fractional_counts[value] * (N_count / total_instances)
        
        subset_entropy = entropy([subset_positive/count, subset_negative/count])
        weighted_entropy += (count / total_instances) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    
    return weighted_entropy, information_gain

path_f = r'C:\Users\Jeremiah\OneDrive - University of Utah\Desktop\Fall 2024\Machine Learning\Github\Machine-Learning\Homework 1\Q3C.csv'
data = pd.read_csv(path_f, index_col=0)

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

total_instances = len(data)
P_count = sum(data['Play?'] == '+')
N_count = total_instances - P_count
root_entropy = entropy([P_count/total_instances, N_count/total_instances])

print("First Stage:")
print(f"Entropy(root) = {root_entropy:.3f}")

infor_gains = {}

for feature in features:
    feature_entropy, ig = cal_feaE_and_IG(data, feature)
    print(f"{feature} (Entropy) = {feature_entropy:.3f} (IG) = {ig:.3f}")
    infor_gains[feature] = ig

root_attribute = max(infor_gains, key=infor_gains.get)
print(f"\nAttribute selected for root: {root_attribute}")

def cal_sub_entropy(subset):
    if len(subset) == 0:
        return 0
    P_count = sum(subset['Play?'] == '+')
    N_count = len(subset) - P_count
    total_count = len(subset)
    return entropy([P_count/total_count, N_count/total_count])

print("\nSecond Stage:")
for value in data[root_attribute].unique():
    if pd.isna(value):
        continue  
    subset = data[data[root_attribute] == value]
    subset_entropy = cal_sub_entropy(subset)
    print(f"{root_attribute} = {value}:")
    print(f"Entropy = {subset_entropy:.3f}")
    
    remaining_features = [f for f in features if f != root_attribute]
    for feature in remaining_features:
        _, ig = cal_feaE_and_IG(subset, feature)
        print(f"  {feature} (IG) = {ig:.3f}")
    print()


# In[97]:


#********Session 2 Question 2*******#

import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, data, attributes, labels, max_depth, benchmark='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.benchmark = benchmark
        self.tree = self.build_tree(data, attributes, labels)

    def build_tree(self, data, attributes, labels, depth=0):
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]
        if len(attributes) == 0 or depth == self.max_depth:
            return np.unique(labels).tolist()[0]
        
        best_attribute = self.choose_attribute(data, labels, attributes)
        tree = {best_attribute: {}}

        for value in set(data[best_attribute]):
            new_data = data[data[best_attribute] == value]
            new_label = labels[data[best_attribute] == value]
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            subtree = self.build_tree(new_data, new_attributes, new_label, depth + 1)
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, data, labels, attributes):
        gains = [self.information_gain(data, labels, attribute) for attribute in attributes]
        return attributes[gains.index(max(gains))]

    def calculate_information_gain(self, data, labels):
        gains = {}
        for attribute in self.attributes:
            gains[attribute] = self.information_gain(data, labels, attribute)
        return gains

    def information_gain(self, data, labels, attribute):
        first_term = self.entropy(labels) if self.benchmark == 'entropy' else \
                     self.gini_index(labels) if self.benchmark == 'gini' else \
                     self.majority_error(labels)

        values, counts = np.unique(data[attribute], return_counts=True)
        weighted_entropy = sum((count / len(data)) * 
                               (self.entropy(labels[data[attribute] == value]) if self.benchmark == 'entropy' else 
                                self.gini_index(labels[data[attribute] == value]) if self.benchmark == 'gini' else 
                                self.majority_error(labels[data[attribute] == value])) 
                               for value, count in zip(values, counts))

        return first_term - weighted_entropy

    def entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        return -np.sum(probabilities * np.log2(probabilities))

    def gini_index(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        return 1 - np.sum(probabilities ** 2)

    def majority_error(self, label):
        _, counts = np.unique(label, return_counts=True)
        return 1 - np.max(counts) / len(label)

    def predict(self, row):
        node = self.tree
        while isinstance(node, dict):
            attribute = list(node.keys())[0]
            attribute_value = row[attribute]
            if attribute_value not in node[attribute]:
                return None
            node = node[attribute][attribute_value]
        return node

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data, label):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)

    def training_error(self, label):
        return self.evaluate(self.data, label)

# Load car data
path_f = r'C:\Users\Jeremiah\Desktop\CS Assignment\car\train.csv'
car_train_data = pd.read_csv(path_f, header=None)
path_j = r'C:\Users\Jeremiah\Desktop\CS Assignment\car\test.csv'
car_test_data = pd.read_csv(path_j, header=None)

car_column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
car_train_data.columns = car_column_names
car_test_data.columns = car_column_names

# Function to display information gain
def display_information_gain(data, labels, dataset_name):
    tree = DecisionTree(data, list(data.columns[:-1]), labels, max_depth=1)
    gains = tree.calculate_information_gain(data, labels)
   
# Calculate and display information gain for car dataset
display_information_gain(car_train_data, car_train_data['label'], "Car Training")
display_information_gain(car_test_data, car_test_data['label'], "Car Test")

# Create dataframe for results
df = pd.DataFrame(columns=["Depth", "Entropy_train", "Entropy_test", "Gini_train", "Gini_test", "Major_train", "Major_test"])

# Train and evaluate decision trees
for depth in range(1, 6):
    row_data = [depth]
    for benchmark in ['entropy', 'gini', 'majority']:
        car_decision_tree = DecisionTree(car_train_data, list(car_train_data.columns[:-1]), car_train_data['label'],
                                         max_depth=depth, benchmark=benchmark)
        
        train_error = car_decision_tree.training_error('label')
        test_error = car_decision_tree.evaluate(car_test_data, 'label')
        
        row_data.extend([train_error, test_error])
        
        print(f"Average prediction training error: {train_error}")
        print(f"Average prediction testing error: {test_error}\n")
    
    df.loc[len(df)] = row_data

# Display results as LaTeX table
print(df.to_latex(index=False, float_format="%.4f"))


# In[106]:


#********Session 2 Question 3*******#


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter

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

# Calculate gini index
def cal_gini_index(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts(normalize=True)
    return 1 - np.sum(label_counts ** 2)

# Calculate majority error
def cal_majority_error(data):
    labels = data.iloc[:, -1]
    majority_class_freq = labels.value_counts(normalize=True).max()
    return 1 - majority_class_freq

# Calculate information gain
def cal_infor_gain(data, attribute, criterion):
    base_impurity = criterion(data)
    attribute_values = data[attribute].unique()
    weighted_impurity = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_impurity += weight * criterion(subset)
    return base_impurity - weighted_impurity

# Choose the best attribute based on the prediction
def ch_best_attribute(data, attributes, criterion):
    criterion_func = {'entropy': cal_entropy, 'gini': cal_gini_index, 'majority': cal_majority_error}[criterion]
    gains = {attr: cal_infor_gain(data, attr, criterion_func) for attr in attributes}
    return max(gains, key=gains.get)

# ID3 Decision Tree algorithm
def id3(data, attributes, criterion, max_depth, current_depth=0):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not attributes or current_depth == max_depth:
        return labels.mode()[0]
    
    best_attribute = ch_best_attribute(data, attributes, criterion)
    tree = {best_attribute: {}}
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            tree[best_attribute][value] = labels.mode()[0]
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            tree[best_attribute][value] = id3(subset, new_attributes, criterion, max_depth, current_depth + 1)
    
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
    
# evaluate the accuracy of the tree
def evaluate(tree, data):
    correct = 0
    for _, row in data.iterrows():
        if predict(tree, row) == row.iloc[-1]:
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

# List of attributes (excluding the label column)
attributes = list(train_data.columns[:-1])

# Initialize the DataFrame for results
df = pd.DataFrame(columns=["Depth", "Entropy_train", "Entropy_test", "Gini_train", "Gini_test", "Major_train", "Major_test"])

# Criteria for splitting and max depths
criteria = ['entropy', 'gini', 'majority']
max_depths = range(1, 17)

# Train and evaluate decision trees
for depth in max_depths:
    row_data = [depth]
    for criterion in criteria:
        tree = id3(train_data, attributes, criterion, depth)
        train_accuracy = evaluate(tree, train_data)
        test_accuracy = evaluate(tree, test_data)
        
        row_data.extend([1 - train_accuracy, 1 - test_accuracy])
    
    df.loc[len(df)] = row_data

    print(f"Depth {depth}:")
    print(f"  Entropy - Train error: {df.iloc[-1]['Entropy_train']:.4f}, Test error: {df.iloc[-1]['Entropy_test']:.4f}")
    print(f"  Gini - Train error: {df.iloc[-1]['Gini_train']:.4f}, Test error: {df.iloc[-1]['Gini_test']:.4f}")
    print(f"  Majority - Train error: {df.iloc[-1]['Major_train']:.4f}, Test error: {df.iloc[-1]['Major_test']:.4f}")


# Generate and print the LaTeX table
latex_table = df.to_latex(index=False, float_format="%.4f")
print("\nLaTeX Table:")
print(latex_table)

# Plot the results
#plt.figure(figsize=(12, 8))
#for criterion in criteria:
 #   if criterion == 'major':
  #      label_prefix = 'Majority'
#    else:
#        label_prefix = criterion.capitalize()
#    plt.plot(df['Depth'], df[f"{criterion.capitalize()}_train"], label=f'Train {label_prefix}')
#    plt.plot(df['Depth'], df[f"{criterion.capitalize()}_test"], label=f'Test {label_prefix}')


# In[ ]:




