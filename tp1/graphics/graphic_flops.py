import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(f"{sys.argv[1]}/{sys.argv[2]}/Flopsn{sys.argv[3]}.csv", header=None)  

df['result'] = df[0] / df[1]

labels = ['0', '1', '2', '3', 'native']

plt.bar(labels, df['result'][:len(labels)])  
plt.xlabel('Muestra')
plt.ylabel('FLOPS')
plt.title(f"FLOPS por muestra, compilador {sys.argv[2]}, tamaño {sys.argv[3]}")

# plt.show()

plt.savefig(f"flops {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png") 
