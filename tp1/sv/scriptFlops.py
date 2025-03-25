import re
import csv
import os
from pprint import pprint
from collections import defaultdict

def parse_file(filename):
    data = defaultdict(lambda: defaultdict(list))
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Dividir el contenido en bloques usando el patrón "Using defaults" como separador
    blocks = re.split(r'Using defaults :', content)[1:]  # Ignorar el primer elemento vacío
    
    for block in blocks:
        try:
            # Extraer N del primer renglón
            N_match = re.search(r'N=(\d+)', block)
            if not N_match:
                continue
            N = N_match.group(1)
            
            # Extraer el compilador
            compiler_match = re.search(r"headless_([^_\s]+)", block)
            if not compiler_match:
                continue
            compiler = compiler_match.group(1)
            
            # Extraer fp_ops
            fp_ops_match = re.search(r'(\d[\d,]+)\s+fp_ret_sse_avx_ops\.all', block)
            if not fp_ops_match:
                continue
            fp_ops = fp_ops_match.group(1).replace(',', '')
            
            # Extraer user time
            user_time_match = re.search(r'(\d+\.\d+)\s+seconds user', block)
            if not user_time_match:
                continue
            user_time = user_time_match.group(1)
            
            # Almacenar los datos
            data[compiler][N].append({
                'fp_ops': fp_ops,
                'user_time': user_time
            })
            
        except Exception as e:
            print(f"Error procesando bloque: {e}")
            print(f"Bloque problemático:\n{block[:200]}...")
            continue
    
    return data

def write_csv_files(data):
    for compiler, n_data in data.items():
        for N, entries in n_data.items():
            output_path = os.path.join(compiler,f'Flopsn{N}.csv')
            file_exists = os.path.exists(output_path)  # Verifica si el archivo ya existe
            with open(output_path, 'a' if file_exists else 'w') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['fp_ops', 'user_time'])
                for entry in entries:
                    writer.writerow([entry['fp_ops'], entry['user_time']])
            print(f"Archivo creado: {output_path}")

def main():
    input_filename = 'nohup.out'  # Cambia esto por tu archivo real
    
    print(f"Procesando archivo: {input_filename}")
    data = parse_file(input_filename)
    
    # Mostrar estadísticas de lo procesado
    pprint(dict(data))
    if data:
        write_csv_files(data)
        print("\nArchivos CSV generados exitosamente!")
    else:
        print("\nNo se encontraron datos válidos en el archivo.")

if __name__ == "__main__":
    main()