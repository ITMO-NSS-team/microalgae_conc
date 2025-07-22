import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Load and process data
def load_data():
    # Load expert data (9 replicates)
    expert_df = pd.read_csv('expert_concentration.csv')
    expert_means = expert_df.iloc[:, 1:-1].mean(axis=1) / 1e6  # Convert to millions
    expert_sd = expert_df.iloc[:, 1:-1].std(axis=1) / 1e6

    # Load auto data (10 replicates)
    auto_df = pd.read_csv('auto_concentration.csv')
    auto_means = auto_df.iloc[:, 1:].mean(axis=1) / 1e6
    auto_sd = auto_df.iloc[:, 1:].std(axis=1) / 1e6

    return pd.DataFrame({
        'Sample': expert_df['Sample'],
        'Expert_Mean': expert_means,
        'Expert_SD': expert_sd,
        'Auto_Mean': auto_means,
        'Auto_SD': auto_sd
    })


# Load high magnification data (from your table)
high_mag_df = pd.DataFrame({
    'Sample': range(1, 22),
    'Expert_Mean': [1.57, 1.89, 3.17, 4.19, 4.81, 5.11, 6.97, 1.00, 11.8, 12.4,
                    12.5, 12.9, 13.0, 13.4, 14.3, 16.7, 17.4, 2.00, 20.2, 30.5, 31.7],
    'Auto_Mean': [1.14, 2.13, 3.29, 3.95, 6.53, 3.65, 5.71, 7.67, 13.7, 14.3,
                  17.3, 14.1, 10.9, 16.4, 14.6, 19.4, 17.6, 15.2, 16.4, 36.2, 39.2]
})

# Create figure
plt.figure(figsize=(9, 4))

# 1. Low Magnification (Distribution Comparison) - Using actual SD from files
low_mag_df = load_data()
plt.subplot(1, 2, 1)
for i, row in low_mag_df.iterrows():
    # Plot point with error bars
    plt.errorbar(row['Expert_Mean'], row['Auto_Mean'],
                 xerr=row['Expert_SD'], yerr=row['Auto_SD'],
                 fmt='o', color='blue', alpha=0.37, capsize=3, elinewidth=1)

    # Add sample number annotation
    plt.text(row['Expert_Mean'] + 1,  # X position (offset slightly right)
             row['Auto_Mean']+2,  # Y position
             str(int(row['Sample'])),  # Sample number
             fontsize=10, color='black', alpha=1,
             va='center', ha='left')

plt.plot([0, 120], [0, 120], 'r--', alpha=0.5, linewidth=1)
plt.xlabel(r'Expert Concentration ($\times 10^6$ cells/mL)', fontsize=12)
plt.ylabel(r'Automatic Concentration ($\times 10^6$ cells/mL)', fontsize=12)
plt.title('A. Low magnification setup', fontsize=14)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0, 125)
plt.ylim(0, 125)

# Calculate and display correlation
r, p = pearsonr(low_mag_df['Expert_Mean'], low_mag_df['Auto_Mean'])
plt.text(5, 110, f'r = {r:.3f}\np = {p:.2e}',
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# 2. High Magnification (Mean Comparison)
plt.subplot(1, 2, 2)
for i, row in high_mag_df.iterrows():
    # Plot point
    plt.scatter(row['Expert_Mean'], row['Auto_Mean'],
                color='blue', alpha=0.37, s=40)
    # Add sample number annotation
    plt.text(row['Expert_Mean'] + 0.3,  # X position (smaller offset)
             row['Auto_Mean'],
             str(int(row['Sample'])),
             fontsize=10, color='black', alpha=1,
             va='center', ha='left')
plt.plot([0, 40], [0, 40], 'r--', alpha=0.5, linewidth=1)
plt.xlabel(r'Expert concentration ($\times 10^6$ cells/mL)', fontsize=12)
plt.title(r'B. High magnification setup', fontsize=14)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0, 42)
plt.ylim(0, 42)

# Calculate and display correlation
r, p = pearsonr(high_mag_df['Expert_Mean'], high_mag_df['Auto_Mean'])
plt.text(2, 37, f'r = {r:.3f}\np = {p:.2e}',
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
plt.show()