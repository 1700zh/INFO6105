import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the processed data
data = pd.read_csv('processed_file.csv')

# Selecting the features for clustering
features = data[['Glucose', 'BMI', 'Age']]

# Normalize the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Assign cluster labels
data['Cluster'] = clusters

# Determine which cluster corresponds to 'Diabetes'
cluster_glucose_means = data.groupby('Cluster')['Glucose'].mean()
diabetes_cluster = cluster_glucose_means.idxmax()  # Cluster with the higher average Glucose

# Map clusters to 'Diabetes' or 'No Diabetes'
data['Outcome'] = data['Cluster'].apply(lambda x: 1 if x == diabetes_cluster else 0)

# Drop the 'Cluster' column as it's no longer needed
data.drop('Cluster', axis=1, inplace=True)

# Save the updated DataFrame
data.to_csv('clustered_file.csv', index=False)
print(data.head())
