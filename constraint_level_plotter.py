import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# Create synthetic data for 2 districts, 3 brands, and 12 months
np.random.seed(42)

# Months (1 to 12)
months = np.arange(1, 14)

# Simulate seasonal sales data for 2 districts and 3 brands
districts = ['District A', 'District B']
brands = ['Brand X', 'Brand Y']

# Create empty dataframe
data = pd.DataFrame(index=months)

# Base seasonal pattern (peaks around months 6-8, troughs around 12 and 3)
seasonal_pattern = 15 * np.sin(2 * np.pi * months / 13) + 50

# Add slight variations specific to each district and brand
if __name__=="__main__":
    for district in districts:
        for brand in brands:
            # Each district/brand has its own level but follows the same seasonal pattern
            level_shift = np.random.uniform(10, 30)  # Different base level for each
            variation = np.random.normal(0, 3, 13)  # Small random noise for variation
            data[f'{district} - {brand}'] = seasonal_pattern + level_shift + variation

    # Concatenate the data for all districts/brands along the x-axis (no gaps)
    full_data = pd.DataFrame()

    # Create an extended index for concatenation (12 months for each district/brand)
    extended_months = np.arange(1, 14)

    # Concatenate the data, keeping continuous flow with color changes
    color_map = plt.cm.Greens  # A color map for smooth transitions
    colors = [color_map(0.7 + 0 * i / (len(districts) * len(brands))) for i in range(len(districts) * len(brands))]

    print(mcolors.to_hex(colors[0]))
    # Initialize the figure
    plt.figure(figsize=(13, 6))

    # Initialize the starting x value (since months are the same for each, no gaps)
    start_month = 1

    # Define the specific month to highlight
    highlight_months = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 34, 47]
    # Loop through columns (sales data for each district and brand)
    for idx, district in enumerate(districts):
        for j, brand in enumerate(brands):
            # Create a shifted x-axis (just stack them)
            x_values = np.arange(start_month, start_month + len(months))

            # Plot the line and fill the area
            plt.plot(x_values, data[f'{district} - {brand}'], label=f'{district} - {brand}',
                     color=colors[idx * len(brands) + j])
            plt.fill_between(x_values, data[f'{district} - {brand}'], color=colors[idx * len(brands) + j], alpha=0.3)

            # Highlight the specific month (highlight the area for the given month)
            for month in highlight_months:
                if month in x_values:
                    if month < 27:
                        clr = "black"
                        ha = '/'
                    else:
                        clr = "darkslategrey"
                        ha = '*'
                    month_idx = np.where(x_values == month)[0][0]  # Find the index of the specified month
                    plt.fill_between(x_values[month_idx:month_idx+2], data[f'{district} - {brand}'][month_idx:month_idx+2], color=clr,
                                alpha=0.7, hatch=ha)
            #
            # highlight_month += 12
            # Update start_month to prevent overlap (shift for next brand/district)
            start_month += len(months)

    # Customize the plot
    plt.yticks([])
    plt.xticks([])
    # plt.xlabel('Month (Across All Districts and Brands)')
    # plt.ylabel('Sales/Revenue')
    # plt.title(f'Sequential Sales/Revenue for Districts and Brands with Highlighted Month {highlight_month}')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()  # To avoid clipping the legend
    plt.show()

