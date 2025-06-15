import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

csv_files = glob.glob(f"{sys.argv[1]}/{sys.argv[2]}/*.csv")

min_dict = {}
for file in csv_files:
    df = pd.read_csv(file)
    mean_value = df.iloc[:, 0].min()  
    min_dict[file] = mean_value

min_df = pd.DataFrame(list(min_dict.items()), columns=['File Name', 'Min'])

min_df_sorted = min_df.sort_values(by='Min', ascending=True)

min_df_sorted['File Name'] = min_df_sorted['File Name'].str.replace(f"{sys.argv[1]}/{sys.argv[2]}/", "").str.replace(".csv", "")

min_df_sorted['Min'] = 1/min_df_sorted['Min']

print("n",sys.argv[3],min_df_sorted)

plt.figure(figsize=(10, 6))  
plt.bar(min_df_sorted['File Name'], min_df_sorted['Min'])
plt.xlabel(f"Optimización")
plt.ylabel("1 / Celdas por nanosegundo")
plt.title(f"Celdas por nanosegundo por optimización, compilador {sys.argv[2]}, tamaño {sys.argv[3]}")
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  

plt.ylim(bottom=0)
# plt.show()
print(f"n {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png")
plt.savefig(f"n/n {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png") 