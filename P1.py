
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Display data
print("Original Data:\n", df.head())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Display processed data
print("\nPreprocessed Data:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
