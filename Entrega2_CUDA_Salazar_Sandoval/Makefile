salida_out = wave

salida_headers = 
salida_source  = $(salida_headers:.h=.c) wave.cu
salida_objects = $(salida_source:.c=.o)

CC     = nvcc
CFLAGS = --compiler-options -Wall -Xptxas="-v"
depends = .depends
build : $(salida_out) 

$(salida_out) : $(salida_objects)
	$(CC)  -o $@ $^  $(CFLAGS)

$(objects) :
	$(CC)  -c -o $@ $*.c $(CFLAGS)

$(depends) : $(salida_source) $(salida_headers)
	@$(CC) -MM $(salida_source) > $@

clean :
	$(RM) $(salida_out)  $(salida_objects) $(zipfile) $(depends) 
.PHONY : build zip clean
sinclude $(depends)
