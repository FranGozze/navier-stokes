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
    
    # Split by compiler declarations and compilation commands
    blocks = re.split(r'(Using compiler: \w+|Compiling .*?\.c with \w+)', content)
    
    for i in range(1, len(blocks), 2):
        block_type = blocks[i].strip()
        block_content = blocks[i+1] if i+1 < len(blocks) else ""
        
        if block_type.startswith('Using compiler:'):
            current_compiler = block_type.split(': ')[1].strip()
            continue
            
        if block_type.startswith('Compiling'):
            # Extract solver name
            solver_match = re.search(r'Compiling (.*?)\.c with', block_type)
            if solver_match:
                current_solver = solver_match.group(1)
            
            # Split the block content into individual runs
            runs = re.split(r'(Running with \d+ threads)', block_content)
            
            for j in range(1, len(runs), 2):
                thread_info = runs[j].strip()
                run_content = runs[j+1] if j+1 < len(runs) else ""
                
                if 'Performance counter stats' not in run_content:
                    continue
                
                try:
                    # Extract thread count
                    thread_match = re.search(r'Running with (\d+) threads', thread_info)
                    thread_count = thread_match.group(1) if thread_match else "1"
                    
                    # Extract N
                    N_match = re.search(r'N=(\d+)', run_content)
                    if not N_match:
                        continue
                    N = N_match.group(1)
                    
                    # Create solver name with thread count
                    solver_name = f"{current_solver}-{thread_count}"
                    
                    # Extract fp_ops
                    fp_ops_match = re.search(r'(\d[\d,]+)\s+fp_ret_sse_avx_ops\.all', run_content)
                    if not fp_ops_match:
                        continue
                    fp_ops = fp_ops_match.group(1).replace(',', '')
                    
                    # Extract elapsed time
                    time_match = re.search(r'(\d+\.\d+)\s+seconds time elapsed', run_content)
                    if not time_match:
                        continue
                    elapsed_time = time_match.group(1)
                    
                    # Store data
                    if current_compiler and current_solver:
                        data[current_compiler][N].append({
                            'solver': solver_name,
                            'fp_ops': fp_ops,
                            'time': elapsed_time,
                        })
                    
                except Exception as e:
                    print(f"Error processing run: {e}")
                    print(f"Problematic run:\n{run_content[:200]}...")
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
                    writer.writerow([entry['solver'], entry['fp_ops'], entry['time']])
            
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