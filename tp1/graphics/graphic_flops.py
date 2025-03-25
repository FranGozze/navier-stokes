import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(f"{sys.argv[1]}/{sys.argv[2]}/Flopsn{sys.argv[3]}.csv", header=None)  

print(f"procesando archivo {sys.argv[1]}/{sys.argv[2]}/Flopsn{sys.argv[3]}.csv")

df['result'] = df[1] / df[2]

labels = df[0].str.strip()  # Get labels from first column and remove whitespace

plt.figure(figsize=(10, 6))  # Set figure size
plt.bar(labels, df['result'])  
plt.xlabel('Optimización')
plt.ylabel('FLOPS')
plt.title(f"FLOPS por optimización, compilador {sys.argv[2]}, tamaño {sys.argv[3]}")

plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff

# plt.show()

plt.savefig(f"flops {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png")