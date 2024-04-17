import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data with clusters
data = pd.read_csv('clustered_file.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert PCA results into DataFrame for further use
train_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
test_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'])

# Add the target variable back to the DataFrames
train_df['Outcome'] = y_train.reset_index(drop=True)
test_df['Outcome'] = y_test.reset_index(drop=True)

# Save the PCA-transformed training and testing sets
train_df.to_csv('train_pca.csv', index=False)
test_df.to_csv('test_pca.csv', index=False)

print("PCA components explained variance ratio:", pca.explained_variance_ratio_)
print(train_df.head())
