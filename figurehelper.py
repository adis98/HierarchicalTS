import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.rand(10) * 10  # Random values for demonstration
})

# Plot with fill
plt.figure(figsize=(8, 5))
plt.plot(df['x'], df['y'], label='Values', color='b')
plt.fill_between(df['x'], df['y'], color='b', alpha=0.3)  # Fill area under the curve

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Filled Area Plot')
plt.legend()
plt.show()