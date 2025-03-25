import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

csv_files = glob.glob(f"{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}/*.csv")

var_dict = {}
for file in csv_files:
    df = pd.read_csv(file)
    var_value = df.iloc[:, 0].var()  
    var_dict[file] = var_value

means_df = pd.DataFrame(list(var_dict.items()), columns=['File Name', 'Mean'])

means_df_sorted = means_df.sort_values(by='Mean', ascending=False)

means_df_sorted['File Name'] = means_df_sorted['File Name'].str.replace(f"{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}/", "").str.replace("../sv/clang/", "").str.replace(".csv", "")

plt.figure(figsize=(10, 6))  
plt.bar(means_df_sorted['File Name'], means_df_sorted['Mean'])
plt.xlabel(f"Optimización")
plt.ylabel("Varianza")
plt.title(f"Varianza de celdas por segundo por optimización, compilador {sys.argv[2]}, tamaño {sys.argv[3]}")
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  

# plt.show()

plt.savefig(f"var {sys.argv[1].replace('../', '')} {sys.argv[2]} {sys.argv[3]}.png") 