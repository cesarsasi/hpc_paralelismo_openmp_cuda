#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <ctype.h>
#include "funciones.h"

// ./wave -N 256 -T 10000 -H 12 -f archivoSalida.raw

int main (int argc, char **argv){

// INICIO: RECOLECCION DE DATOS INICIALES
    // Variables
    int c;
    char * charGrilla;
    int tamanoGrilla;
    char * charNPasos;
    int numPasos;
    char * charNHebras;
    int numHebras;
    char * archivoSalida;
    int parametros = 0;
    // Obtener datos ingresados en la instruccion en consola mediante getOpt, para ser almacenados respectivamente
    while ((c = getopt (argc, argv, "N:T:H:f:")) != -1){
        switch (c){
            // Almacena y setea el dato dependiendo del valor entregado por parametro.
            case 'N':
                charGrilla= optarg;
                tamanoGrilla = atoi(charGrilla);
                parametros ++;
                break;
            case 'T':
                charNPasos = optarg;
                numPasos = atoi(charNPasos);
                parametros ++;
                break;
            case 'H':
                charNHebras = optarg;
                numHebras = atoi(charNHebras);
                parametros ++;
                break;    
            case 'f':
                archivoSalida = optarg;
                parametros ++;
                break;
            case '?':
                //Si es que se ingresó un parámetro que existe pero no se entregó un argumento requerido
                if(optopt == 'N' || optopt == 'T' || optopt == 'H' || optopt == 'f'){
                    fprintf (stderr, "El parametro -%c no puede estar vacio.\n", optopt);
                //Si es que la opción ingresada no existe
                } else if (isprint (optopt)){ 
                    fprintf (stderr, "El parametro -%c no es requerido. \n", optopt);
                }
            default:
                abort ();
        }       
    }
    //Se validan los parámetros obligatorias, validando si fueron 5
    if(parametros != 4){ 
        printf("Se deben ingresar los siguientes parametros: \n-N <Tamano de la grilla>\n-T <Numero de Pasos>\n-H <Numero de hebras>\n-f <Archivo de salida>\n");
        return 0;
    }
    //Se validan los parámetros de entrada
    if(validar_entrada(tamanoGrilla, numPasos, numHebras) == 0){
        return 0;
    }
// FIN: RECOLECCION DE DATOS INICIALES

    //Asigna memoria para matriz transitoria para tiempo N-2 que representa la matriz anterior a la anterior.
    float *matriz_t0 = malloc(tamanoGrilla*tamanoGrilla*sizeof(float));
    //Asigna memoria para matriz transitoria para tiempo N-1 que representa la matriz anterior.
    float *matriz_t1 = malloc(tamanoGrilla*tamanoGrilla*sizeof(float));
    //Matriz N que representa la matriz actual de calculo.
    float *matriz = malloc(tamanoGrilla*tamanoGrilla*sizeof(float));
    
// INICIO: SOLUCION SECUENCIAL
    double inicio_s, termino_s;
    inicio_s = omp_get_wtime();
    //Se inicializan todas las matrices con ceros para evitar que contengan basura
    matriz_ceros(matriz_t0,tamanoGrilla);
    matriz_ceros(matriz_t1,tamanoGrilla);
    matriz_ceros(matriz,tamanoGrilla);
    for (int t = 0; t < numPasos; t++){
        ejecutar_schroedinger(matriz_t0, matriz_t1, matriz, tamanoGrilla, t);
    }
    termino_s = omp_get_wtime();
    //imprime resultados
    printf("Ejecucion secuencial: %f (segundos)\n", termino_s - inicio_s);
    FILE *salida = fopen("archivoSalida_secuncial.raw", "w");
    fwrite(matriz, sizeof(float), tamanoGrilla*tamanoGrilla, salida);
    fclose(salida);
// FIN: SOLUCION SECUENCIAL
    

// INICIO: SOLUCION PARALELA 
    double inicio_p, termino_p;
    inicio_p = omp_get_wtime();
    matriz_ceros_paralelo(matriz_t0,tamanoGrilla,numHebras);
    matriz_ceros_paralelo(matriz_t1,tamanoGrilla, numHebras);
    matriz_ceros_paralelo(matriz,tamanoGrilla, numHebras);
    for (int t = 0; t < numPasos; t++){
        ejecutar_schroedinger_paralelo(matriz_t0, matriz_t1, matriz, tamanoGrilla, t, numHebras);
    }
    termino_p = omp_get_wtime();
    //imprime resultados
    printf("Ejecucion paralela: %f (segundos)\n", termino_p - inicio_p);
    FILE *salida2 = fopen(archivoSalida, "w");
    fwrite(matriz, sizeof(float), tamanoGrilla*tamanoGrilla, salida2);
    fclose(salida2);
// FIN: SOLUCION PARALELA

    //Se libera toda la memoria reservada utilizada.
    free(matriz_t0);
    free(matriz_t1);
    free(matriz);
    return 0;
}
