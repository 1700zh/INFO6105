import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the PCA-transformed data
train_df = pd.read_csv('train_pca.csv')
test_df = pd.read_csv('test_pca.csv')

# Prepare training and test sets
X_train, y_train = train_df.drop('Outcome', axis=1), train_df['Outcome']
X_test, y_test = test_df.drop('Outcome', axis=1), test_df['Outcome']

# Define base classifiers with their own hyperparameters
base_classifiers = {
    'GaussianNB': GaussianNB(),
    'MLPClassifier': MLPClassifier(max_iter=2000, early_stopping=True),
    'KNeighborsClassifier': KNeighborsClassifier()
}

# Hyperparameters for each classifier
params = {
    'GaussianNB': {'var_smoothing': np.logspace(0, -9, num=100)},
    'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]},
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

# Train each base classifier using GridSearchCV with 5-fold CV and store best parameters
best_params = {}
base_predictions = []  # Initializing the list to store predictions for the meta learner

for name in base_classifiers:
    grid = GridSearchCV(base_classifiers[name], params[name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_params[name] = grid.best_params_
    print(f"Best parameters for {name}: {grid.best_params_}")
    # Predict on training set for meta learner training
    pred = grid.predict(X_train)
    base_predictions.append(pred.reshape(-1, 1))

# Combine predictions for training the meta learner
X_meta_train = np.hstack(base_predictions)

# Meta learner: Decision Tree
meta_learner = DecisionTreeClassifier()
param_grid_meta = {'max_depth': [3, 5, 7, 10], 'criterion': ['gini', 'entropy']}
grid_meta = GridSearchCV(meta_learner, param_grid_meta, cv=5, scoring='accuracy')
grid_meta.fit(X_meta_train, y_train)
print(f"Best parameters for meta learner: {grid_meta.best_params_}")

# Predict using base classifiers on test set using their best parameters
test_base_predictions = []

for name in base_classifiers:
    # Set the best parameters found for each classifier
    base_classifiers[name].set_params(**best_params[name])
    base_classifiers[name].fit(X_train, y_train)  # Retrain on whole train set
    pred = base_classifiers[name].predict(X_test)
    test_base_predictions.append(pred.reshape(-1, 1))

# Combine test set predictions
X_meta_test = np.hstack(test_base_predictions)

# Predict on test set using meta learner
meta_learner.set_params(**grid_meta.best_params_)
meta_learner.fit(X_meta_train, y_train)  # Retrain on whole meta train set
final_predictions = meta_learner.predict(X_meta_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy of the super learner model on the test data: {accuracy:.4f}")
