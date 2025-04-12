import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

csv_files = glob.glob(f"{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}/*.csv")

means_dict = {}
for file in csv_files:
    df = pd.read_csv(file)
    mean_value = df.iloc[:, 0].mean()  
    means_dict[file] = mean_value

means_df = pd.DataFrame(list(means_dict.items()), columns=['File Name', 'Mean'])

means_df_sorted = means_df.sort_values(by='Mean', ascending=False)

means_df_sorted['File Name'] = means_df_sorted['File Name'].str.replace(f"{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}/", "").str.replace("../sv/clang/", "").str.replace(".csv", "")

print("n",sys.argv[3],means_df_sorted)

plt.figure(figsize=(10, 6))  
plt.bar(means_df_sorted['File Name'], means_df_sorted['Mean'])
plt.xlabel(f"Optimización")
plt.ylabel("Promedio de cálculo de celdas por nanosegundo")
plt.title(f"Celdas por nanosegundo por optimización, compilador {sys.argv[2]}, tamaño {sys.argv[3]}")
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  

plt.ylim(bottom=0)
# plt.show()
print(f"n {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png")
plt.savefig(f"n/n {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png") 