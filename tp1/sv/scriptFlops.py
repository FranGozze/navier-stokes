# Para ejecutar el script, primero se debe crear el archivo aux.txt con el contenido del compilador que nos interese obtener. Esto se obtiene de ver en el archivo "nohup.out" del servidor. Lo unico que tenes que poner en "aux.txt" son las lineas que contengan el compilador. Mejorar para automatizar esto.

import re
import os  # <- Añade esta línea

def extract_data(input_file, output_folder='output'):
    # Diccionario para guardar los resultados por N
    results = {'128': [], '256': [], '512': []}
    
    with open(input_file, 'r') as f:
        current_block = []
        current_n = None
        
        for line in f:
            line = line.strip()
            
            # Detectamos el inicio de un nuevo bloque (línea con N=)
            if line.startswith('Using defaults : N='):
                if current_n and current_block:
                    results[current_n].append(current_block)
                current_n = line.split('N=')[1].split()[0]
                current_block = []
            
            # Extraemos el valor antes de fp_ret_sse_avx_ops.all
            elif 'fp_ret_sse_avx_ops.all' in line:
                fp_value = line.split()[0]
                current_block.append(fp_value.replace(',', ''))
            
            # Extraemos el valor antes de seconds user
            elif 'seconds user' in line:
                user_value = line.split()[0]
                current_block.append(user_value)
        
        # Añadimos el último bloque
        if current_n and current_block:
            results[current_n].append(current_block)
    
    # Escribimos los archivos CSV
    for n, data in results.items():
        output_path = os.path.join(output_folder,f'Flopsn{n}.csv')
        with open(output_path, 'w') as f_out:
            # f_out.write("fp_operations,user_seconds\n")
            for row in data:
                if len(row) == 2:  # Aseguramos que tenemos ambos valores
                    f_out.write(f"{row[0]},{row[1]}\n")

# Uso del script
extract_data("aux.txt","icx")