import re
import csv
import os
from pprint import pprint
from collections import defaultdict

def parse_file(filename):
    data = defaultdict(lambda: defaultdict(list))
    current_name = None
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Dividir por bloques que comienzan con "Name:" seguido de "Using defaults"
    blocks = re.split(r'(Name: ".*?")\n', content)
    
    # Los bloques impares son los nombres, los pares son el contenido
    for i in range(1, len(blocks), 2):
        if i+1 >= len(blocks):
            break
            
        # Extraer el nombre
        name_match = re.search(r'Name: "(.*?)"', blocks[i])
        if not name_match:
            continue
        current_name = name_match.group(1)
        
        # Procesar el bloque de rendimiento asociado
        performance_blocks = re.split(r'Using defaults :', blocks[i+1])
        
        for perf_block in performance_blocks[1:]:  # Saltar el primer elemento vacío
            try:
                # Extraer N
                N_match = re.search(r'N=(\d+)', perf_block)
                if not N_match:
                    continue
                N = N_match.group(1)
                
                # Extraer el compilador
                compiler_match = re.search(r"headless_([^_\s]+)", perf_block)
                if not compiler_match:
                    continue
                compiler = compiler_match.group(1)
                
                # Extraer fp_ops
                fp_ops_match = re.search(r'(\d[\d,]+)\s+fp_ret_sse_avx_ops\.all', perf_block)
                if not fp_ops_match:
                    continue
                fp_ops = fp_ops_match.group(1).replace(',', '')
                
                # Extraer user time
                user_time_match = re.search(r'(\d+\.\d+)\s+seconds user', perf_block)
                if not user_time_match:
                    continue
                user_time = user_time_match.group(1)
                
                # Almacenar los datos
                data[compiler][N].append({
                    'name': current_name,
                    'fp_ops': fp_ops,
                    'user_time': user_time
                })
                
            except Exception as e:
                print(f"Error procesando bloque: {e}")
                print(f"Bloque problemático:\n{perf_block[:200]}...")
                continue
    
    return data

def write_csv_files(data):
    for compiler, n_data in data.items():
        # Crear directorio si no existe
        os.makedirs(compiler, exist_ok=True)
        
        for N, entries in n_data.items():
            output_path = os.path.join(compiler, f'Flopsn{N}.csv')
            file_exists = os.path.exists(output_path)
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Escribir encabezado solo si el archivo es nuevo
                if not file_exists:
                    writer.writerow(['name', 'fp_ops', 'user_time'])
                
                for entry in entries:
                    writer.writerow([entry['name'], entry['fp_ops'], entry['user_time']])
            
            print(f"Archivo {'actualizado' if file_exists else 'creado'}: {output_path}")

def main():
    input_filename = 'nohup.out'  # Asegúrate de que este es el nombre correcto
    
    print(f"Procesando archivo: {input_filename}")
    data = parse_file(input_filename)
    
    # Mostrar estadísticas de lo procesado
    pprint(dict(data))
    if data:
        write_csv_files(data)
        print("\nArchivos CSV generados exitosamente!")
    else:
        print("\nNo se encontraron datos válidos en el archivo.")
        print("Verifica que el archivo de entrada tenga el formato correcto.")

if __name__ == "__main__":
    main()