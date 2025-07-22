import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties

# Load data
auto_df = pd.read_csv('auto_time.csv')
expert_df = pd.read_csv('expert_time.csv')

# Merge data
merged = pd.merge(auto_df, expert_df, on='Sample', suffixes=('_auto', '_expert'))

# Create figure with adjusted layout
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.7])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax_table = fig.add_subplot(gs[2])

# Plot 1: Time vs Concentration (Log-Log)
sc_expert = ax1.scatter(merged['Concentration'], merged['Time, s_expert'],
                        c='red', alpha=0.7, s=80, label='Expert')
sc_auto = ax1.scatter(merged['Concentration'], merged['Time, s_auto'],
                      c='green', alpha=0.7, s=80, label='Auto')

# Add sample annotations
for i, row in merged.iterrows():
    ax1.annotate(f"{int(row['Sample'])}",
                 xy=(row['Concentration'], row['Time, s_expert']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, alpha=0.8)
    ax1.annotate(f"{int(row['Sample'])}",
                 xy=(row['Concentration'], row['Time, s_auto']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, alpha=0.8)


# Logarithmic trendline
def log_func(x, a, b):
    return a * np.log(x) + b



popt, _ = curve_fit(log_func, merged['Concentration'], merged['Time, s_expert'])
x_fit = np.linspace(min(merged['Concentration']), max(merged['Concentration']), 100)
logs = log_func(x_fit, *popt)
logs[0] = 10
ax1.plot(x_fit, logs, 'r--',
         label=f'Expert Trend')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Cell Concentration (cells/mL)', fontsize=12)
ax1.set_ylabel('Processing Time (seconds)', fontsize=12)
ax1.set_title('Time Efficiency Comparison', fontsize=14)
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.3)

# Plot 2: Bar chart
ax2.bar(['Expert', 'Auto'],
        [merged['Time, s_expert'].mean(), merged['Time, s_auto'].mean()],
        color=['red', 'green'], alpha=0.7)
ax2.set_ylabel('Average Time (seconds)', fontsize=12)
ax2.set_title('Mean Processing Time', fontsize=14)
ax2.text(0, merged['Time, s_expert'].mean(), f'{merged["Time, s_expert"].mean():.0f}s',
         ha='center', va='bottom', fontsize=10)
ax2.text(1, merged['Time, s_auto'].mean(), f'{merged["Time, s_auto"].mean():.1f}s',
         ha='center', va='bottom', fontsize=10)

# Create data table
table_data = merged[['Sample', 'Concentration', 'Time, s_expert', 'Time, s_auto']]
table_data['Speedup'] = (merged['Time, s_expert'] / merged['Time, s_auto']).round(1)
table_data['Sample'] = table_data['Sample'].astype(str)
table_data['Concentration'] = round(table_data['Concentration'] / 1000000, 2)
table_data['Time, s_auto'] = table_data['Time, s_auto'].astype(int)

table_data.columns = ['Sample', f'Conc.\n(10⁶)', 'Expert', 'Auto', 'Speedup']

ax_table.axis('off')
table = ax_table.table(cellText=table_data.values,
                       colLabels=table_data.columns,
                       loc='center',
                       cellLoc='center',
                       colColours=['#f0f0f0'] * 5)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.5, 1.8)

# Make header bold
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_text_props(fontproperties=FontProperties(weight='bold', size=9))

# Add title to table
ax_table.set_title('Measurement Data', fontsize=12, pad=20)

# Add overall title
plt.suptitle('Automated vs Manual Cell Counting Efficiency', fontsize=16)

plt.tight_layout()
plt.savefig(f'time_compare.png', dpi=400)
plt.show()
