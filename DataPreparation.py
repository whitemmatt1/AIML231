from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from functools import partial

import pandas as pd
import numpy as np
import random as random

# added libraries 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Please do not change these random seeds.
np.random.seed(231)
random.seed(231)
model = KNeighborsClassifier(n_neighbors=5)

# --------------------------- Part 1: Load the Data ---------------------------

data_train = pd.read_csv('CodeAndData/Data/Training.csv')
X_train, y_train = data_train.drop('Status', axis=1), data_train['Status']

data_test = pd.read_csv('CodeAndData/Data/Test.csv')
X_test, y_test = data_test.drop('Status', axis=1), data_test['Status']

# --------------------------- Part 1: Data Preprocessing ---------------------------

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  # List of numerical feature names
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()  # List of categorical feature names

print(numerical_cols)
print(categorical_cols)
### Step 1: Handle Missing Values ###
# Define and apply appropriate imputers for numerical and categorical features.
# Ensure consistency between X_train and X_test.

num_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

### Step 2: Encoding Categorical Features ###
# Convert categorical features into numerical format using encoding techniques.

encoder = OneHotEncoder(sparse_output=False)

train_idx = X_train.index
test_idx = X_test.index

X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=train_idx)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test_idx)

X_train_dropped = X_train.drop(columns=categorical_cols)
X_test_dropped = X_test.drop(columns=categorical_cols)

X_train_processed = pd.concat([X_train_dropped, X_train_encoded], axis=1)
X_test_processed = pd.concat([X_test_dropped, X_test_encoded], axis=1)

### Step 3: Feature Scaling ###
# Apply scaling/normalization to features.

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns, index=X_train_processed.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_processed.columns, index=X_test_processed.index)


X_train_process = X_train_scaled  # Processed/transformed training set
X_test_process = X_test_scaled  # Processed/transformed test set

# Train the model using all available features.
model.fit(X_train_process, y_train)
y_pred = model.predict(X_test_process)
print(f"All Features: {balanced_accuracy_score(y_test, y_pred):.4f}")

# --------------------------- Part 2: Feature Ranking ---------------------------
# Use SelectKBest with mutual_info_classif to select the top 7 features from the processed training set X_train_process.
# You should use mutual_info_fix as the parameter for SelectKBest to ensure random_state is set to 231 for reproducibility.
mutual_info_fix = partial(mutual_info_classif, random_state=231)
selector_kbest = SelectKBest(score_func=mutual_info_fix, k=7)  # Replace [] with SelectKBest implementation
X_train_kbest = selector_kbest.fit_transform(X_train_process, y_train)  # Replace [] with transformed training data
X_test_kbest = selector_kbest.transform(X_test_process)  # Replace [] with transformed test data

feature_names = X_train_process.columns
selected_features = feature_names[selector_kbest.get_support()]
print("Selected features:", selected_features.tolist())

# Train the model with selected features.
model.fit(X_train_kbest, y_train)
y_pred = model.predict(X_test_kbest)
print(f"Feature Ranking: {balanced_accuracy_score(y_test, y_pred):.4f}")

# extracting selected features from dataframe
selected_features = feature_names[selector_kbest.get_support()].tolist()
top_features_df = X_train_process[selected_features]

# corr matrix for heatmap
corr_matrix = top_features_df.corr()

# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            square=True, fmt='.2f', linewidths=0.5)
plt.title("Pearson Correlation Between Top Seven Features")
plt.tight_layout()
plt.show()
# --------------------------- Part 3: Sequential Feature Selection ---------------------------
# Use Sequential Backward Selection (SBS) to select a subset of 7 features from the processed training set X_train_process.
clf = KNeighborsClassifier(n_neighbors=5)
selector_sequential = SequentialFeatureSelector(estimator=clf, n_features_to_select=17, direction='backward')  # Replace [] with SequentialFeatureSelector implementation
X_train_sbfs = selector_sequential.fit_transform(X_train_processed, y_train)  # Replace [] with transformed training data
X_test_sbfs = selector_sequential.transform(X_test_processed)  # Replace [] with transformed test data

# Train the model with selected features.
model.fit(X_train_sbfs, y_train)
y_pred = model.predict(X_test_sbfs)
print(f"Sequential Feature Selection: {balanced_accuracy_score(y_test, y_pred):.4f}")

# getting the selected feature names
selected_features_mask = selector_sequential.get_support()
selected_feature_names = X_train_processed.columns[selected_features_mask].tolist()
print("SBS selected features:", selected_feature_names)

# extracting selected features from dataframe
selected_features_df = X_train_processed[selected_feature_names]

# corr matrix for heatmap
corr_matrix = selected_features_df.corr()

# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            square=True, fmt='.2f', linewidths=0.5)
plt.title("Pearson Correlation Between Seven Selected Features")
plt.tight_layout()
plt.show()

# line plot for 6.2a
feature_counts = [1, 3, 7, 12, 17]
accuracies = [0.5240, 0.5853, 0.5697, 0.5697, 0.5697]

plt.figure(figsize=(10, 6))
plt.plot(feature_counts, accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Number of Features')
plt.ylabel('Balanced Accuracy')
plt.title('KNN Performance vs Number of Features Selected by SBS')
plt.grid(True)
plt.xticks(feature_counts)
plt.show()