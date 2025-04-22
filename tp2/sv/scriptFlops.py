import re
import csv
import os
from pprint import pprint
from collections import defaultdict

def parse_file(filename):
    data = defaultdict(lambda: defaultdict(list))
    current_compiler = None
    current_solver = None
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Dividir por bloques que comienzan con "Using compiler:" o "Compiling..."
    blocks = re.split(r'(Using compiler: \w+|Compiling .*?\.c with \w+)', content)
    
    for i in range(1, len(blocks), 2):
        block_type = blocks[i].strip()
        block_content = blocks[i+1] if i+1 < len(blocks) else ""
        
        if block_type.startswith('Using compiler:'):
            current_compiler = block_type.split(': ')[1].strip()
            continue
            
        if block_type.startswith('Compiling'):
            # Extraer el nombre del solver (ej: "solver_vector1.c")
            solver_match = re.search(r'Compiling (.*?)\.c with', block_type)
            if solver_match:
                current_solver = solver_match.group(1)
            
            # Procesar el bloque de rendimiento si existe
            if block_content and 'Performance counter stats' in block_content:
                try:
                    # Extraer N
                    N_match = re.search(r'N=(\d+)', block_content)
                    if not N_match:
                        continue
                    N = N_match.group(1)
                    
                    # Extraer fp_ops
                    fp_ops_match = re.search(r'(\d[\d,]+)\s+fp_ret_sse_avx_ops\.all', block_content)
                    if not fp_ops_match:
                        continue
                    fp_ops = fp_ops_match.group(1).replace(',', '')
                    
                    # Extraer user time
                    user_time_match = re.search(r'(\d+\.\d+)\s+seconds user', block_content)
                    if not user_time_match:
                        continue
                    user_time = user_time_match.group(1)
                    
                    # Almacenar los datos
                    if current_compiler and current_solver:
                        data[current_compiler][N].append({
                            'solver': current_solver,
                            'fp_ops': fp_ops,
                            'user_time': user_time
                        })
                    
                except Exception as e:
                    print(f"Error procesando bloque: {e}")
                    print(f"Bloque problemático:\n{block_content[:200]}...")
                    continue
    
    return data

def write_csv_files(data):
    for compiler, n_data in data.items():
        # Crear directorio si no existe
        os.makedirs(compiler, exist_ok=True)
        
        for N, entries in n_data.items():
            output_path = os.path.join(compiler, f'Flopsn{N}.csv')
            
            # Siempre sobrescribir el archivo
            with open(output_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Escribir encabezado
                # writer.writerow(['solver', 'fp_ops', 'user_time'])
                
                for entry in entries:
                    writer.writerow([entry['solver'], entry['fp_ops'], entry['user_time']])
            
            print(f"Archivo creado/actualizado: {output_path}")

def main():
    input_filename = '../../resultados_run_all'  # Asegúrate que este es el nombre correcto
    
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