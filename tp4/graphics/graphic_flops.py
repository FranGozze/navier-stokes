import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys
import os

# Create output directory if it doesn't exist
os.makedirs("flops", exist_ok=True)

# Read all CSV files for the given size
size = sys.argv[3]
compilers = ['gcc', 'clang', 'icx']
data = {}

for compiler in compilers:
    file_path = f"{sys.argv[1]}/{compiler}/Flopsn{size}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)
        df['result'] = df[1] / df[2]
        data[compiler] = df

# Get all unique optimization labels and their max values
opt_max_values = {}
for label in set().union(*[set(df[0].str.strip()) for df in data.values()]):
    max_value = 0
    for df in data.values():
        if label in df[0].values:
            value = df[df[0].str.strip() == label]['result'].values[0]
            max_value = max(max_value, value)
    opt_max_values[label] = max_value

# Sort labels by their maximum values
labels = sorted(opt_max_values.keys(), key=lambda x: opt_max_values[x], reverse=True)

# Calculate min_flops before creating the plot
all_values = []
for df in data.values():
    all_values.extend(df['result'].values)
max_flops = max(all_values)
min_flops = min(all_values)
max_speedup = max_flops / min_flops

# Calculate appropriate figure height
base_height = 6
label_space = max_flops * 0.15  # Increased space for labels
total_height = base_height + (label_space / max_flops) * base_height

# Create the plot
plt.figure(figsize=(12, total_height))

# Set width of bars and positions of the bars
barWidth = 0.25
r1 = range(len(labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars
for i, (compiler, df) in enumerate(data.items()):
    positions = [r1, r2, r3][i]
    values = []
    for label in labels:
        if label in df[0].values:
            value = df[df[0].str.strip() == label]['result'].values[0]
        else:
            value = 0
        values.append(value)
    bars = plt.bar(positions, values, width=barWidth, label=compiler)
    
    # Add speedup text above each bar
    for bar, value in zip(bars, values):
        if value > 0:  # Only show speedup for non-zero values
            speedup = value / min_flops
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_flops * 0.02,
                    f'{speedup:.2f}x',
                    ha='center', va='bottom', rotation=90)

# Add labels and title
plt.xlabel('Optimización')
plt.ylabel('FLOPS')
plt.title(f"FLOPS por optimización y compilador, tamaño {size} (ordenado por máximo FLOPS)")

# Set y-axis limits to include space for labels
plt.ylim(0, max_flops * 1.2)  # Add 20% more space at the top

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(labels))], labels, rotation=45, ha='right')

# Add legend
plt.legend()

# Adjust layout with more padding at the top
plt.tight_layout(pad=2.0)

# Display the highest speedup ratio in the corner
plt.text(0.02, 0.98, f'{max_speedup:.2f}x', 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8))

# Save the plot
plt.savefig(f"flops/flops {sys.argv[1].replace('../', '')} grouped {size}.png")