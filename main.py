import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
df = pd.read_csv('data.csv')

# Calculate the frequency of each store name
stores_freq = df['name'].value_counts()

# Define an exponential function for fitting
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Normalize the x-values
x = np.arange(len(stores_freq))
x_norm = x / max(x)

y = stores_freq.values

# Fit the exponential function to the data using normalized x-values
params, covariance = curve_fit(exp_func, x_norm, y)

# Plot the bar chart
ax = stores_freq.plot.bar(title='Frequency of Stores in Rhein-Neckar', width=1.2, color='skyblue', figsize=(12, 6))
plt.ylim(0, 62)
plt.xlabel('Store Name (every 50 displayed)')
plt.ylabel('Frequency')

# Plot the exponential trend line using normalized x-values
plt.plot(x, exp_func(x_norm, *params), color='red', linestyle='dotted', label='trend line')
plt.legend()

# Selectively show labels
ax.set_xticks(ax.get_xticks()[::50])  # Show every 50th label
ax.set_xticklabels(stores_freq.index[::50], rotation=45, fontsize=8)  # Rotate labels, set fontsize

plt.tight_layout()
plt.show()
