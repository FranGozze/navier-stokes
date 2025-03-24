# Compiladores disponibles
COMPILERS=gcc clang icx icc

# Flags adicionales
EXTRA_FLAGS=-O0 -O1 -O2 -O3

# Lista de valores numéricos para ejecutar
VALUES=128 256 512

# Compilador por defecto
CC=gcc

# Flags por defecto
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -march=native
LDFLAGS=

# Sufijo para el nombre del ejecutable
SUFFIXX=
SUFFIX=

SOURCES=$(shell echo *.c)
# Objetos comunes
COMMON_OBJECTS=solver.o wtime.o

# Objetivo principal
all: $(foreach compiler,$(COMPILERS),$(foreach flags,$(EXTRA_FLAGS),build_$(compiler)_$(flags)))

# Regla para construir el ejecutable con diferentes compiladores y flags
build_%:
	$(eval compiler=$(word 1,$(subst _, ,$*)))
	$(eval flags=$(word 2,$(subst _, ,$*)))
	@echo "Building with $(compiler) and flags $(flags)"
	$(MAKE) CC=$(compiler) CFLAGS="$(CFLAGS) $(flags)" SUFFIX=_$(compiler)_$(flags) headless



demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut


# Regla para compilar el objetivo headless
headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o headless$(SUFFIX) $(LDFLAGS)

# Regla para ejecutar el ejecutable con diferentes valores y guardar los resultados en archivos
run: all
	$(foreach compiler,$(COMPILERS),\
		$(foreach flags,$(EXTRA_FLAGS),\
			$(foreach value,$(VALUES),\
				perf stat -e fp_ret_sse_avx_ops.all ./headless_$(compiler)_$(flags) $(value) \
					 > tp1/sv/$(compiler)/n$(value)$(flags)$(SUFFIXX).csv;)))

# Limpieza
clean:
	rm -f $(TARGETS) *.o .depend *~ headless_* output_*.txt

# Dependencias
.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all build_% run
