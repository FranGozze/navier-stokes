CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter
SUFFIX= -01
LDFLAGS=


TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=solver.o wtime.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

hdTest: headless.o $(COMMON_OBJECTS) 
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	./hdTest 128 > tp1/n128$(SUFFIX).csv
	./hdTest 256 > tp1/n256$(SUFFIX).csv
	./hdTest 512 > tp1/n512$(SUFFIX).csv

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
