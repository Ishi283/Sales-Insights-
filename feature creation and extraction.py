# DATA PREPROCESSING - SALES INSIGHTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from google.colab import files

# Load the dataset
final_df = pd.read_csv("sales_insights_merged.csv")


# 1. Average Price Per Item

final_df["Avg_Price_Per_Item"] = (
    final_df["Sales_Amount"]
    / final_df["Quantity"].replace(0, np.nan)
)

final_df["Avg_Price_Per_Item"] = (
    final_df["Avg_Price_Per_Item"].fillna(0)
)


# 2. Profit Percentage

final_df["Profit_Percentage"] = (
    final_df["Profit"]
    / final_df["Sales_Amount"].replace(0, np.nan)
) * 100

final_df["Profit_Percentage"] = (
    final_df["Profit_Percentage"].fillna(0)
)


# 3. High Sales

median_sales = final_df["Sales_Amount"].median()

final_df["High_Sales"] = (
    final_df["Sales_Amount"] > median_sales
).astype(int)


# Display newly created features

print(final_df[
    ["Avg_Price_Per_Item",
     "Profit_Percentage",
     "High_Sales"]
].head())


# Statistical summary

print(final_df.describe())


# Sales Amount Distribution

plt.hist(final_df["Sales_Amount"], bins=15)

plt.xlabel("Sales Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Sales Amount")

plt.show()


# Sales Amount vs High Sales

sns.boxplot(
    data=final_df,
    x="High_Sales",
    y="Sales_Amount"
)

plt.title("Sales Amount vs High Sales")
plt.show()


# Correlation Heatmap

numeric_df = final_df.select_dtypes(include="number")

correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=False,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()