#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

#include <getopt.h>

#include <sys/time.h>
#include <time.h>

// FUNCIÓN: validar_entrada.
// ENTRADA: Parametros de entrada con valor entero a excepcion de archivoSalida.
// PROCESAMIENTO: Valida los parametros de entrada, con tal de asegurar un correcto input de variables.
// SALIDA: Valor entero, con posibilidad de "0" si son entradas invalidas y un "1" en caso contrario.
int validar_entrada(int tamanoGrila, int numPasos, int tamanoBloqueX, int tamanoBloqueY){
    int valido = 1;
    
    if(tamanoGrila == 0){
        printf("Ingrese un número de grillas válido.\n");
        valido = 0; 
    }
    if(numPasos == 0){
        printf("Ingrese un número de pasos válido.\n");
        valido = 0;
    }
    if(tamanoBloqueX == 0){
        printf("Ingrese un tamano bloque X válido.\n");
        valido = 0;
    }
    if(tamanoBloqueY == 0){
        printf("Ingrese un tamano bloque Y válido.\n");
        valido = 0;
    }
    return valido;
}

// FUNCIÓN: Matriz_ceros.
// ENTRADA: Matriz inicializada
// PROCESAMIENTO: Rellena toda la matriz con ceros.
// SALIDA: Matriz de ceros.
void matriz_ceros(float *matriz, int N){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            matriz[i*N+j] = 0.0;
        }
    }
}

// FUNCIÓN: Ejecutar_schroedinger_t0cuda
// ENTRADA: Matriz tiempo 0, largo del arreglo.
// PROCESAMIENTO: Calcula primer pulso de forma paralela.
// SALIDA: - .
__global__ void ejecutar_schroedinger_t0cuda(float *matriz_t0, int N){
	int i, j;
	i = blockDim.x*blockIdx.x + threadIdx.x;  // global index x (horizontal)
	j = blockDim.y*blockIdx.y + threadIdx.y;  // global index y (vertical)

	if((i > 0.4*N && i <0.6*N) && (j > 0.4*N && j < 0.6*N)){
        matriz_t0[i*N+j] = 20.0;
	}
}
 
// FUNCIÓN: Ejecutar_schroedinger_t1cuda
// ENTRADA: Matriz tiempo 0, Matriz tiempo 1, largo del arreglo.
// PROCESAMIENTO: Calcula tiempo 1 de forma paralela.
// SALIDA: - .
__global__ void ejecutar_schroedinger_t1cuda(float *matriz_t0, float *matriz_t1, int N){
	int i, j;
    float c = 1.0, dt = 0.1, dd = 2.0;
    float iInf, iSup, jInf, jSup, ij_t1, cola;
	i = blockDim.x*blockIdx.x + threadIdx.x;  // global index x (horizontal)
	j = blockDim.y*blockIdx.y + threadIdx.y;  // global index y (vertical)

    if(i >= 1 && i < (N-1) && j >= 1 && j < (N-1)){
        iInf = matriz_t0[(i-1)*N+j];
        iSup = matriz_t0[(i+1)*N+j];
        jInf = matriz_t0[i*N+(j-1)];
        jSup = matriz_t0[i*N+(j+1)];
        ij_t1 = matriz_t0[i*N+j];
        cola = ((dt*dt)/(dd*dd)) * (iInf+iSup+jSup+jInf-(4*ij_t1));
        matriz_t1[i*N+j] = (ij_t1 + (((c*c)/2)*cola));
	}              
}

// FUNCIÓN: Ejecutar_schroedinger_tncuda
// ENTRADA: Matriz tiempo 0, Matriz tiempo 1, Matriz tiempo actual,largo del arreglo.
// PROCESAMIENTO: Calcula tiempo n de forma paralela.
// SALIDA: - .
__global__ void ejecutar_schroedinger_tncuda(float *matriz_t0, float *matriz_t1, float *matriz_rs, int N){
	int i, j;
    float c = 1.0, dt = 0.1, dd = 2.0;
    float iInf, iSup, jInf, jSup, ij_t1, ij_t0, cola;
	i = blockDim.x*blockIdx.x + threadIdx.x;  // global index x (horizontal)
	j = blockDim.y*blockIdx.y + threadIdx.y;  // global index y (vertical)

    if(i >= 1 && i < (N-1) && j >= 1 && j < (N-1)){
        iInf = matriz_t1[(i-1)*N+j];
        iSup = matriz_t1[(i+1)*N+j];
        jInf = matriz_t1[i*N+(j-1)];
        jSup = matriz_t1[i*N+(j+1)];
        ij_t1 = matriz_t1[i*N+j];
        ij_t0 = matriz_t0[i*N+j];
        cola = ((dt*dt)/(dd*dd)) * (iInf+iSup+jSup+jInf-(4*ij_t1));
        matriz_rs[i*N+j] = ((2*ij_t1) - ij_t0 + ((c*c)*cola));
	}                 
}

// FUNCIÓN: Copiar_matriz_cuda
// ENTRADA: Matriz tiempo 0, Matriz tiempo 1, Matriz tiempo actual,largo del arreglo.
// PROCESAMIENTO: Copia los valores de las matrices a la matriz del tiempo anterior tn -> tn-1 -> tn-2.
// SALIDA: - .
__global__ void copiar_matriz_cuda(float *matriz_t0, float *matriz_t1, float *matriz_rs, int N){
    int i, j;
	i = blockDim.x*blockIdx.x + threadIdx.x;  // global index x (horizontal)
	j = blockDim.y*blockIdx.y + threadIdx.y;  // global index y (vertical)

    if(i >= 1 && i < N-1 && j >= 1 && j < N-1){
        matriz_t0[i*N+j] = matriz_t1[i*N+j];
        matriz_t1[i*N+j] = matriz_rs[i*N+j];
    }
}


// BLOQUE PRINCIPAL
    // ./wave -N 256 -T 10000 -H 12 -f archivoSalida.raw
__host__ int main (int argc, char **argv){

// INICIO: RECOLECCION DE DATOS INICIALES
    // Variables
    int c, tamanoGrilla, tamanoBloqueX, tamanoBloqueY, numPasos;
    char *charGrilla, *charBloqueX, *charBloqueY, *charNPasos, *archivoSalida;
    int parametros = 0;
    // Obtener datos ingresados en la instruccion en consola mediante getOpt, para ser almacenados respectivamente
    while ((c = getopt (argc, argv, "N:x:y:T:f:")) != -1){
        switch (c){
            // Almacena y setea el dato dependiendo del valor entregado por parametro.
            case 'N':
                charGrilla= optarg;
                tamanoGrilla = atoi(charGrilla);
                parametros ++;
                break;
            case 'x':
                charBloqueX = optarg;
                tamanoBloqueX = atoi(charBloqueX);
                parametros ++;
                break;
            case 'y':
                charBloqueY = optarg;
                tamanoBloqueY = atoi(charBloqueY);
                parametros ++;
                break;
            case 'T':
                charNPasos = optarg;
                numPasos = atoi(charNPasos);
                parametros ++;
                break; 
            case 'f':
                archivoSalida = optarg;
                parametros ++;
                break;
            case '?':
                //Si es que se ingresó un parámetro que existe pero no se entregó un argumento requerido
                if(optopt == 'N' || optopt == 'x' || optopt == 'y' || optopt == 'T' || optopt == 'f'){
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
    if(parametros != 5){ 
        printf("Se deben ingresar los siguientes parametros: \n-N <Tamano de la grilla>\n-T <Numero de Pasos>\n-H <Numero de hebras>\n-f <Archivo de salida>\n");
        return 0;
    }
    //Se validan los parámetros de entrada
    if(validar_entrada(tamanoGrilla, numPasos, tamanoBloqueX, tamanoBloqueY ) == 0){
        return 0;
    }
// FIN: RECOLECCION DE DATOS INICIALES

    //Pedir memoria al host, para las matrices (CPU)
    int tamArreglo = tamanoGrilla*tamanoGrilla;
    float *h_matriz_t0 = (float *)malloc(tamArreglo*sizeof(float));
    float *h_matriz_t1 = (float *)malloc(tamArreglo*sizeof(float));
    float *h_matriz_rs = (float *)malloc(tamArreglo*sizeof(float));
    //Inicializar valores para cada matriz del host
    matriz_ceros(h_matriz_t0,tamanoGrilla);
    matriz_ceros(h_matriz_t1,tamanoGrilla);
    matriz_ceros(h_matriz_rs,tamanoGrilla);

// INICIO: SOLUCION CUDA   
    //Se empieza a medir el tiempo
    cudaEvent_t tInicioGpu, tFinGpu;
    float tiempoTranscurrido;
    cudaEventCreate(&tInicioGpu);
    cudaEventCreate(&tFinGpu);
    cudaEventRecord(tInicioGpu,0);
    time_t tInicio = time(NULL);

    //Numero de threads en cada bloquewarpsTotales
    dim3 tamBloque, tamGrilla;
    tamGrilla.x = (int)ceil((float)tamanoGrilla/tamanoBloqueX);
    tamGrilla.y = (int)ceil((float)tamanoGrilla/tamanoBloqueY);
    //Numero de bloques en la grilla
    tamBloque.x = tamanoBloqueX;
    tamBloque.y = tamanoBloqueY;

    int numBloques = tamGrilla.x*tamGrilla.y;
    int TamBloque = tamanoBloqueX*tamanoBloqueY;

    //Pedir memoria al device, para las matrices (GPU)
    float *d_matriz_t0, *d_matriz_t1, *d_matriz_rs;
    cudaMalloc(&d_matriz_t0, tamArreglo*sizeof(float));
    cudaMalloc(&d_matriz_t1, tamArreglo*sizeof(float));
    cudaMalloc(&d_matriz_rs, tamArreglo*sizeof(float));
    //Se copia la matriz del host a la matriz del devicetInicioGpu
    cudaMemcpy(d_matriz_t0, h_matriz_t0, tamArreglo*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matriz_t1, h_matriz_t1, tamArreglo*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matriz_rs, h_matriz_rs, tamArreglo*sizeof(float), cudaMemcpyHostToDevice);

    ejecutar_schroedinger_t0cuda<<<tamGrilla,tamBloque>>>(d_matriz_t0, tamanoGrilla);
    cudaMemcpy(h_matriz_t0, d_matriz_t0, tamArreglo*sizeof(float), cudaMemcpyDeviceToHost);
   
    ejecutar_schroedinger_t1cuda<<<tamGrilla,tamBloque>>>(d_matriz_t0, d_matriz_t1, tamanoGrilla);
    cudaMemcpy(h_matriz_t1, d_matriz_t1, tamArreglo*sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 2; i <= numPasos; i++){
        ejecutar_schroedinger_tncuda<<<tamGrilla,tamBloque>>>(d_matriz_t0, d_matriz_t1, d_matriz_rs, tamanoGrilla);
        copiar_matriz_cuda<<<tamGrilla,tamBloque>>>(d_matriz_t0, d_matriz_t1, d_matriz_rs, tamanoGrilla);
    }
    cudaMemcpy(h_matriz_t1, d_matriz_t1, tamArreglo*sizeof(float), cudaMemcpyDeviceToHost);
// FIN: SOLUCION CUDA

    //Se termina de medir el tiempo
    time_t tFin = time(NULL);
    cudaEventRecord(tFinGpu,0);
    cudaEventSynchronize(tFinGpu);
    cudaEventElapsedTime( &tiempoTranscurrido, tInicioGpu, tFinGpu);
    printf("Tiempo (wall-clock)  : %f seg.\n", (float)(tFin-tInicio));
    printf("Tiempo transcurrido  : %f seg.\n", tiempoTranscurrido/1000);

    //Warps Utilizados
    int device, warpsActivos, warpsTotales;
    //Obtener propiedades de la GPU
    cudaDeviceProp propiedadesDevice;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&propiedadesDevice, device);
    //Porcentaje de warps utilizados
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBloques, ejecutar_schroedinger_t0cuda, TamBloque, 0);
    warpsActivos = numBloques*TamBloque /propiedadesDevice.warpSize;
    warpsTotales = propiedadesDevice.maxThreadsPerMultiProcessor/propiedadesDevice.warpSize;
    printf ("Warps totales        : %d uds.\n", warpsTotales);
    printf ("Warps activos        : %d uds.\n", warpsActivos);
    printf ("Porcentaje utilizado : %f .\n", (float)warpsActivos/warpsTotales*100);

    //Se almacena la imagen
    FILE *salida2 = fopen(archivoSalida, "w");
    fwrite(h_matriz_t1, sizeof(float), tamArreglo, salida2);
    fclose(salida2);

    //Se libera toda la memoria reservada utilizada.
    cudaEventDestroy(tInicioGpu);
    cudaEventDestroy(tFinGpu);
    cudaFree(d_matriz_t0);
    cudaFree(d_matriz_t1);
    cudaFree(d_matriz_rs);
    free(h_matriz_t0);
    free(h_matriz_t1);
    free(h_matriz_rs);
    return 0;
}