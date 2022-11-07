#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "funciones.h"

//VARIABLES GLOBALES

float c = 1.0;  // Constante que representa la velocidad de la onda en el medio,
float dt = 0.1; // Constante que representa el intervalo de tiempo con que avanza la simulacion.
float dd = 2.0; // Constante que representa el ambio en la superficie.

/*  FUNCIÓN: validar_entrada.
    ENTRADA: Parametros de entrada con valor entero a excepcion de archivoSalida.
    PROCESAMIENTO: Valida los parametros de entrada, con tal de asegurar un correcto input de variables.
    SALIDA: Valor entero, con posibilidad de "0" si son entradas invalidas y un "1" en caso contrario.
*/
int validar_entrada(int tamanoGrila, int numPasos, int numHebras){
    int valido = 1;
    
    if(tamanoGrila == 0){
        printf("Ingrese un número de grillas válido.\n");
        valido = 0; 
    }
    if(numPasos == 0){
        printf("Ingrese un número de pasos válido.\n");
        valido = 0;
    }
    if(numHebras == 0){
        printf("Ingrese una cantidad de hebras válida.\n");
        valido = 0;
    }
    return valido;
}

/*  FUNCIÓN: copiar_matriz.
    ENTRADA: void
    PROCESAMIENTO: Copia una matriz.
    SALIDA: void
*/
void copiar_matriz(float *matriz_origen,float *matriz_destino, int N){
  for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            matriz_destino[i*N+j] = matriz_origen[i*N+j];
        }
    }
}


/*  FUNCIÓN: Matriz_ceros.
    ENTRADA: Matriz inicializada
    PROCESAMIENTO: Rellena toda la matriz con ceros.
    SALIDA: Matriz de ceros.
*/
int matriz_ceros(float *matriz, int N){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            matriz[i*N+j] = 0.0;
        }
    }
    return 0;
}

/*  FUNCIÓN: ejecutar_schroedinger.
    ENTRADA: Tres matrices que representan los estados durante los tiempos t-1, t-2, el actual. 
            Largo de la matriz y el paso a ejecutar
    PROCESAMIENTO: Rellena toda la matriz con ceros.
    SALIDA: Matriz de ceros.
*/
int ejecutar_schroedinger(float *matriz_t0, float *matriz_t1, float *matriz, int N, int t){
    //este for recorrera cada celda e irá modificando las matrices según corresponda.
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){    
            //se desencadena el primer pulso recordando que la matriz ya fue inicializada con ceros.
            if(t == 0){
                if((i > 0.4*N && i <0.6*N) && (j > 0.4*N && j < 0.6*N))
                    matriz_t1[i*N+j] = 20.0;
            }else{
                if(i == 0 || j == 0 || i == N-1 || j == N-1)
                    matriz_t1[i*N+j] = 0.0;
                else{
                    float iInf, iSup, jInf, jSup, ij_t1, cola;
                    iInf = matriz_t1[(i-1)*N+j];
                    iSup = matriz_t1[(i+1)*N+j];
                    jInf = matriz_t1[i*N+(j-1)];
                    jSup = matriz_t1[i*N+(j+1)];
                    ij_t1 = matriz_t1[i*N+j];
                    cola = ((dt*dt)/(dd*dd)) * (iInf+iSup+jSup+jInf-(4*ij_t1));
                    if(t == 1){
                        matriz[i*N+j] = (ij_t1 + (((c*c)/2)*cola));
                    }else {
                        float ij_t0 = matriz_t0[i*N+j];
                        matriz[i*N+j] = ((2*ij_t1) - ij_t0 + ((c*c)*cola));
                    }
                }
            }   
        }   
    }  
    //Se copian las matrices, inicialmente el primer pulso se guarda en t1 por lo que se debe copiar a t0.   
    copiar_matriz(matriz_t1, matriz_t0, N);
    if(t > 0)
        copiar_matriz(matriz, matriz_t1, N);      
    return 0;
}


/*  FUNCIÓN: copiar_matriz.
    ENTRADA: void
    PROCESAMIENTO: Copia una matriz.
    SALIDA: void
*/
void copiar_matriz_paralelo(float *matriz_origen,float *matriz_destino, int N, int numHebras){
    #pragma omp parallel shared(matriz_origen, matriz_destino) num_threads(numHebras)
    {
        //collapse: se utiliza para convertir un bucle anidado en un solo bucle que será paralelizado.
        #pragma omp for collapse(2)
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                matriz_destino[i*N+j] = matriz_origen[i*N+j];
            }
        }
    }
}


/*  FUNCIÓN: Matriz_ceros.
    ENTRADA: Matriz inicializada
    PROCESAMIENTO: Rellena toda la matriz con ceros.
    SALIDA: Matriz de ceros.
*/
int matriz_ceros_paralelo(float *matriz, int N, int numHebras){
    #pragma omp parallel shared(matriz) num_threads(numHebras)
    {
        //collapse: se utiliza para convertir un bucle anidado en un solo bucle que será paralelizado.
        #pragma omp for
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                matriz[i*N+j] = 0.0;
            }
        }
    }
    return 0;
}

/*  FUNCIÓN: ejecutar_schroedinger.
    ENTRADA: Tres matrices que representan los estados durante los tiempos t-1, t-2, el actual. Largo de la matriz y el paso a ejecutar
    PROCESAMIENTO: Rellena toda la matriz con ceros.
    SALIDA: Matriz de ceros.
*/
int ejecutar_schroedinger_paralelo(float *matriz_t0, float *matriz_t1, float *matriz, int N, int t, int numHebras){
    //este for recorrera cada celda e irá modificando las matrices según corresponda.
    //El tiempo es necesariamente una variable secuencial, no se puede paralelizar debido a que debe esperar que los dos calculos anteriores se realicen con éxito.
    #pragma omp parallel shared(matriz_t0, matriz_t1, matriz) num_threads(numHebras)
    {
        //collapse: se utiliza para convertir un bucle anidado en un solo bucle que será paralelizado.
        #pragma omp for
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){    
                // Se desencadena el primer pulso recordando que la matriz ya fue inicializada con ceros.
                // El pulso inicial también se ejecuta de forma paralela.
                if(t == 0){
                    if((i > 0.4*N && i <0.6*N) && (j > 0.4*N && j < 0.6*N))
                        // #pragma omp critical
                        matriz_t1[i*N+j] = 20.0;
                }else{
                    if(i == 0 || j == 0 || i == N-1 || j == N-1)
                        // #pragma omp critical
                        matriz_t1[i*N+j] = 0.0;
                    else{
                        float iInf, iSup, jInf, jSup, ij_t1, cola;
                        iInf = matriz_t1[(i-1)*N+j];
                        iSup = matriz_t1[(i+1)*N+j];
                        jInf = matriz_t1[i*N+(j-1)];
                        jSup = matriz_t1[i*N+(j+1)];
                        ij_t1 = matriz_t1[i*N+j];
                        cola = ((dt*dt)/(dd*dd)) * (iInf+iSup+jSup+jInf-(4*ij_t1));
                        //El caso critico t=1 solo difiere de los otros en 2 aspectos, pero solo se ejecutará una vez.
                        if(t == 1){
                            // #pragma omp critical
                            matriz[i*N+j] = (ij_t1 + (((c*c)/2)*cola));
                        //La mayoría de las iteraciones contemplaran este caso.
                        }else {
                            float ij_t0 = matriz_t0[i*N+j];
                            // #pragma omp critical no es necesario ya que cada hebra escribirá una celda única.
                            //desde el t>1 ocurre SIMD. 
                            matriz[i*N+j] = ((2*ij_t1) - ij_t0 + ((c*c)*cola));
                        }
                    }
                }   
            }   
        } 
    }
    //Se copian las matrices, inicialmente el primer pulso se guarda en t1 por lo que se debe copiar a t0.   
    copiar_matriz_paralelo(matriz_t1, matriz_t0, N, numHebras);
    if(t > 0)
        copiar_matriz_paralelo(matriz, matriz_t1, N, numHebras);      
    return 0;
}

