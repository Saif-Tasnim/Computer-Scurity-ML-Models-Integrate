import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

# Step 1: Read the original dataset
df = pd.read_csv('../Datasets/transaction - used.csv')

# Step 2: Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42)

# Step 3: Reduce the shuffled dataset to 20k entries
df_sampled = df_shuffled.sample(n=20000)

# Step 4: Preprocess the data
X = df_sampled.drop('isFraud', axis=1)
y = df_sampled['isFraud']

# Encode categorical variables
X = pd.get_dummies(X, columns=['type'])

# Step 5: Plot a bar chart for the 'isFraud' column
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar')
plt.title('Distribution of isFraud')
plt.xlabel('isFraud')
plt.ylabel('Count')
plt.show()

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Step 8: Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Step 9: Train the model and measure the time taken
start_time = time.time()
nb_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Step 10: Predict the target variable for the training, validation, and testing sets
y_train_pred = nb_model.predict(X_train)
y_val_pred = nb_model.predict(X_val)
y_test_pred = nb_model.predict(X_test)

# Step 11: Print the output of training, validation, and testing sets
print("Training set accuracy:", (y_train_pred == y_train).mean())
print("Validation set accuracy:", (y_val_pred == y_val).mean())
print("Testing set accuracy:", (y_test_pred == y_test).mean())

# Step 12: Print the total time taken for training, validation, and testing
print("Training time:", train_time)
