import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

# Load the data
data_path = '/Users/cornerback30/Desktop/algorithm/q7-q8-q9.xlsx'
df = pd.read_excel(data_path)

# Prepare the data for linear regression
X = np.array([7, 8, 9]).reshape(-1, 1)

# Store results
results = []

# Directory to save the plots
plots_dir = '../results/plots/'
os.makedirs(plots_dir, exist_ok=True)

# Apply linear regression for each customer
for customer_id in df['Müşteri Id'].unique():
    customer_data = df[df['Müşteri Id'] == customer_id]
    y = customer_data[['q7', 'q8', 'q9']].values.flatten()
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the slope (coefficient) of the regression line
    slope = model.coef_[0]
    
    # Determine trend
    if slope < -30:
        trend = 'Potential Churn'
    else:
        trend = 'Business as Usual'
    
    results.append({
        'Müşteri Id': customer_id,
        'Slope': slope,
        'Trend': trend
    })

    # Plotting
    plt.figure()
    plt.plot([7, 8, 9], y, marker='o', label='Actual')
    plt.plot([7, 8, 9], model.predict(X), linestyle='--', label='Predicted')
    plt.xlabel('Quarter')
    plt.ylabel('Segment Score')
    plt.title(f'Customer {customer_id} - Trend: {trend}')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(plots_dir, f'customer_{customer_id}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved for customer {customer_id} at {plot_path}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
output_path = '../results/customer_trends.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_excel(output_path, index=False)

print(f'Results saved to {output_path}')