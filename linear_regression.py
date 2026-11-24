# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
final_df = pd.read_csv("sales_insights_merged.csv")

# Select independent and dependent variables
X = final_df["Marketing_Spend"]
Y = final_df["Sales_Amount"]

# Remove rows containing missing values in either selected column
data = pd.DataFrame({
    "X": X,
    "Y": Y
}).dropna()

X = data["X"]
Y = data["Y"]

n = len(data)

print("Number of observations:", n)

# STEP 1: Calculate means
X_mean = X.mean()
Y_mean = Y.mean()

print("\nSTEP 1: MEANS")
print("Mean of X:", X_mean)
print("Mean of Y:", Y_mean)

# STEP 2: Calculate deviations
data["X_deviation"] = X - X_mean
data["Y_deviation"] = Y - Y_mean

# STEP 3: Calculate product of deviations
data["Product"] = (
    data["X_deviation"] * data["Y_deviation"]
)

# STEP 4: Calculate squared deviations of X
data["X_squared_deviation"] = (
    data["X_deviation"] ** 2
)

print("\nCALCULATION TABLE")
print(data.head(10))

# STEP 5: Calculate sums
sum_products = data["Product"].sum()
sum_squared_X = data["X_squared_deviation"].sum()

print("\nSTEP 5: SUMMATIONS")
print("Sum of products:", sum_products)
print("Sum of squared X deviations:", sum_squared_X)

# STEP 6: Calculate slope
b1 = sum_products / sum_squared_X

print("\nSTEP 6: SLOPE")
print("b1 =", b1)

# STEP 7: Calculate intercept
b0 = Y_mean - (b1 * X_mean)

print("\nSTEP 7: INTERCEPT")
print("b0 =", b0)

# STEP 8: Construct regression equation
print("\nSTEP 8: REGRESSION EQUATION")
print(f"Predicted Sales Amount = {b0:.4f} + {b1:.4f} * Marketing Spend")

# STEP 9: Calculate predicted values
data["Predicted_Y"] = b0 + b1 * X

# STEP 10: Calculate residuals
data["Residual"] = Y - data["Predicted_Y"]

print("\nSTEP 9 AND 10: PREDICTIONS AND RESIDUALS")
print(data[["X", "Y", "Predicted_Y", "Residual"]].head(10))

# STEP 11: Calculate SSE
data["Squared_Error"] = data["Residual"] ** 2
SSE = data["Squared_Error"].sum()

print("\nSTEP 11: SSE")
print("SSE =", SSE)

# STEP 12: Calculate SST
data["Squared_Total_Deviation"] = (Y - Y_mean) ** 2
SST = data["Squared_Total_Deviation"].sum()

# Calculate R-squared
R2 = 1 - (SSE / SST)

print("\nSTEP 12: GOODNESS OF FIT")
print("SST =", SST)
print("R-squared =", R2)
print("R-squared percentage =", R2 * 100, "%")

# STEP 13: Verify using sklearn
model = LinearRegression()
model.fit(X.to_frame(), Y)

print("\nSKLEARN VERIFICATION")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R-squared:", model.score(X.to_frame(), Y))

# STEP 14: Plot scatter plot and regression line
plt.figure(figsize=(8, 6))

plt.scatter(
    X, Y,
    label="Actual observations"
)

plt.plot(
    X,
    data["Predicted_Y"],
    color="red",
    label="Regression line"
)

plt.xlabel("Marketing Spend")
plt.ylabel("Sales Amount")
plt.title("Simple Linear Regression: Marketing Spend vs Sales")
plt.legend()
plt.grid(True)

plt.show()