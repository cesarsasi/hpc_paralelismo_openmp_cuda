salida_out = wave


salida_headers = wave.h funciones.h
salida_source  = $(salida_headers:.h=.c) wave.c funciones.c
salida_objects = $(salida_source:.c=.o)

CC     = gcc
CFLAGS = -Wall -fopenmp

depends = .depends

build : $(salida_out) 

$(salida_out) : $(salida_objects)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(depends) : $(salida_source) $(salida_headers)
	@$(CC) -MM $(salida_source) > $@ -lm


clean :
	$(RM) $(salida_out) $(zipfile) $(depends)

.PHONY : build zip clean

sinclude $(depends)
