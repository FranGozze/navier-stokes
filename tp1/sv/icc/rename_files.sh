#!/bin/bash

# Encuentra todos los archivos que contienen '--' en su nombre
find . -name "*--*" -type f | while read -r file; do
    # Obtiene el nombre del archivo sin la ruta
    filename=$(basename "$file")
    # Reemplaza '--' por '-' en el nombre
    newname="${filename//--/-}"
    # Si el nombre cambió, renombra el archivo
    if [ "$filename" != "$newname" ]; then
        # Obtiene el directorio del archivo
        dir=$(dirname "$file")
        # Renombra el archivo
        mv -v "$file" "$dir/$newname"
    fi
done
