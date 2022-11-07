#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int validar_entrada(int tamanoGrila, int numPasos, int numHebras);

void copiar_matriz(float *matriz_origen, float *matriz_destino, int N);

void copiar_matriz_paralelo(float *matriz_origen, float *matriz_destino, int N, int numHebras);

int matriz_ceros(float *matriz, int N);

int matriz_ceros_paralelo(float *matriz, int N, int numHebras);

int ejecutar_schroedinger(float *matriz_t0, float *matriz_t1, float *matriz, int N, int t);

int ejecutar_schroedinger_paralelo(float *matriz_t0, float *matriz_t1, float *matriz, int N, int t, int numHebras);


