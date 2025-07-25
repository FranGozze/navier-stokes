# Compiladores disponibles
COMPILERS=gcc

# Flags adicionales
EXTRA_FLAGS=-O1

# Lista de valores numéricos para ejecutar
VALUES=128

# Compilador por defecto
CC=gcc

# Flags por defecto
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -march=native -funsafe-math-optimizations -fopt-info -ftree-vectorize -fopenmp
LDFLAGS=

# Sufijo para el nombre del ejecutable
SUFFIXX=
SUFFIX=

# Objetos comunes (now with suffix)
COMMON_OBJECTS=solver$(SUFFIX).o wtime$(SUFFIX).o

# Objetivo principal
all: $(foreach compiler,$(COMPILERS),$(foreach flags,$(EXTRA_FLAGS),build_$(compiler)_$(flags)))

# Regla para construir el ejecutable con diferentes compiladores y flags
build_%:
	$(eval compiler=$(word 1,$(subst _, ,$*)))
	$(eval flags=$(word 2,$(subst _, ,$*)))
	@echo "Building with $(compiler) and flags $(flags)"
	$(MAKE) headless CC=$(compiler) CFLAGS="$(CFLAGS) $(flags)" SUFFIX=_$(compiler)_$(flags)

# Regla para compilar el objetivo headless
headless: headless$(SUFFIX).o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o headless$(SUFFIX) $(LDFLAGS)

# Pattern rule for object files with suffix
%.o:
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rules for suffixed object files
solver$(SUFFIX).o: solver.c
	$(CC) $(CFLAGS) -c $< -o $@

wtime$(SUFFIX).o: wtime.c
	$(CC) $(CFLAGS) -c $< -o $@

headless$(SUFFIX).o: headless.c
	$(CC) $(CFLAGS) -c $< -o $@

# Regla para ejecutar el ejecutable con diferentes valores y guardar los resultados en archivos
run: all
	$(foreach compiler,$(COMPILERS),\
		$(foreach flags,$(EXTRA_FLAGS),\
			$(foreach value,$(VALUES),\
				perf stat -e fp_ret_sse_avx_ops.all -e cycles -e cpu-clock -e task-clock -- ./headless_$(compiler)_$(flags) $(value) \
					 > tp2/sv/$(compiler)/n$(value)$(flags)$(SUFFIXX).csv;)))

demo.o: demo.c
	$(CC) $(CFLAGS) -c $< -o $@

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

# Limpieza
clean:
	rm -f $(TARGETS) *.o .depend *~ headless_* output*.txt demo

# Dependencias
.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all build_% run