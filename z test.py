# TWO-SAMPLE Z-TEST
# Comparing Sales Amount of Business and Retail Customers

import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest

# Load dataset
df = pd.read_csv('sales_insights_merged(1).csv')

# Create the two groups
business_sales = df[df['Customer_Type'] == 'Business']['Sales_Amount']
retail_sales = df[df['Customer_Type'] == 'Retail']['Sales_Amount']

# Display basic information
print("Business Customers:")
print("Sample Size:", len(business_sales))
print("Mean Sales:", business_sales.mean())

print("\nRetail Customers:")
print("Sample Size:", len(retail_sales))
print("Mean Sales:", retail_sales.mean())

# Perform Two-Sample Z-Test
z_stat, p_value = ztest(
    business_sales,
    retail_sales,
    value=0
)

print("\n----- Z-Test Results -----")
print("Z-Statistic:", z_stat)
print("P-Value:", p_value)

# Significance level
alpha = 0.05

# Decision
if p_value < alpha:
    print("\nReject the Null Hypothesis (H0)")
    print("There is a statistically significant difference")
    print("between the mean sales of Business and Retail customers.")
else:
    print("\nFail to Reject the Null Hypothesis (H0)")
    print("There is no statistically significant difference")
    print("between the mean sales of Business and Retail customers.")