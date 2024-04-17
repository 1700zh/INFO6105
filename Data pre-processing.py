import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('diabetes_project.csv')

# Function to remove outliers
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Remove outliers
data_clean = remove_outliers(data)

# Impute missing values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data_clean), columns=data_clean.columns)

# Normalize the data
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

# Output the cleaned data
data_normalized.to_csv('processed_file.csv', index=False)
print(data_normalized.head())
