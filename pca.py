# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the dataset
final_df = pd.read_csv("sales_insights_merged(1).csv")


# Select numerical features for PCA
features = [
    "Customer_Age",
    "Unit_Price",
    "Quantity",
    "Discount_Percent",
    "Cost_Per_Unit",
    "Marketing_Spend",
    "Customer_Rating",
    "Delivery_Days",
    "Sales_Amount",
    "Profit"
]

X = final_df[features]


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply PCA
pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)


# Add Principal Components to the dataset
final_df["PC1"] = pca_data[:, 0]
final_df["PC2"] = pca_data[:, 1]


# Display PCA values
print(final_df[["PC1", "PC2"]].head())


# Display explained variance
print("Explained Variance:")
print(pca.explained_variance_ratio_)


# Plot PCA
plt.figure(figsize=(8, 6))

plt.scatter(
    final_df["PC1"],
    final_df["PC2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Sales Dataset")

plt.show()