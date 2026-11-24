# DATA PREPROCESSING - SALES INSIGHTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from google.colab import files

# 1. Upload and load dataset
uploaded = files.upload()
df = pd.read_excel('sales_insight_raw_data_100_with_missing_values.xlsx')

print("Original Dataset:")
print(df.head())
print("Shape:", df.shape)

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Fill missing values
num_cols = ['Customer_Age', 'Discount_Percent',
            'Marketing_Spend', 'Customer_Rating', 'Delivery_Days']

cat_cols = ['Customer_Type', 'City', 'Payment_Method']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Remove duplicate records
print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()

# 5. Convert data types
df['Date'] = pd.to_datetime(df['Date'])

# 6. Check invalid values
print("\nInvalid Ages:", (df['Customer_Age'] <= 0).sum())
print("Invalid Quantity:", (df['Quantity'] <= 0).sum())
print("Invalid Discount:",
      ((df['Discount_Percent'] < 0) |
       (df['Discount_Percent'] > 100)).sum())

# 7. Detect outliers using IQR
outlier_cols = ['Unit_Price', 'Quantity',
                'Marketing_Spend', 'Delivery_Days']

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(col, "Outliers:",
          ((df[col] < lower) | (df[col] > upper)).sum())

    # Outlier capping
    df[col] = df[col].clip(lower, upper)

# 8. Categorical encoding
df_encoded = pd.get_dummies(
    df,
    columns=['Customer_Type', 'City', 'Region',
             'Product_Category', 'Payment_Method'],
    dtype=int
)

# 9. Feature scaling
scale_cols = ['Customer_Age', 'Unit_Price', 'Quantity',
              'Discount_Percent', 'Cost_Per_Unit',
              'Marketing_Spend', 'Customer_Rating',
              'Delivery_Days']

scaler = StandardScaler()

df_scaled = df_encoded.copy()
df_scaled[scale_cols] = scaler.fit_transform(
    df_scaled[scale_cols]
)

# 10. Final results
print("\nCleaned Dataset:")
print(df.head())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nFinal Dataset Shape:", df.shape)
print("Encoded Dataset Shape:", df_encoded.shape)
print("Scaled Dataset Shape:", df_scaled.shape)

# 11. Save preprocessed dataset
df.to_csv('sales_insights_cleaned.csv', index=False)
df_encoded.to_csv('sales_insights_encoded.csv', index=False)
df_scaled.to_csv('sales_insights_scaled.csv', index=False)

files.download('sales_insights_cleaned.csv')

print("\nData Preprocessing Completed Successfully!")