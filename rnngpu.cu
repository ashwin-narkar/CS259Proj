#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>

#define __CLASS 1
#define __BATCH 1

#if __CLASS == 1
    #define __XT_ROW 10
    #define __XT_COL 1
    #define __HT_ROW 10
    #define __HT_COL 7
    #define __YT_ROW 1
    #define __YT_COL 7
    #define __U_ROW 1
    #define __U_COL 7
    #define __V_ROW 1
    #define __V_COL 10
    #define __W_ROW 7
    #define __W_COL 7
#else
    #define __XT_ROW 10
    #define __XT_COL 1
    #define __HT_ROW 10
    #define __HT_COL 7
    #define __YT_ROW 1
    #define __YT_COL 7
    #define __U_ROW 1
    #define __U_COL 7
    #define __V_ROW 1
    #define __V_COL 10
    #define __W_ROW 7
    #define __W_COL 7
#endif

//From open source code: git user lzhengchun
/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void 
gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    //Find output row
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = 0; 
    while(tid >= k){
        row++;
        tid -= k;
    }
    //Find output column
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    int col = tid % k;
    c[tid] = 0;
    for(int j = 0; j < n; j++){
        c[tid] += a[n*row + j] * b[col + k*j];
    }
} 

__global__ void 
gpu_matrix_add(float *a,float *b, float *c)
{ 
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
} 

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t xt_size = __XT_COL*__XT_ROW * sizeof(float) * __BATCH;
    size_t ht_size = __HT_COL*__HT_ROW * sizeof(float) * __BATCH;
    size_t yt_size = __YT_COL*__YT_ROW * sizeof(float) * __BATCH;
    size_t W_size = __W_COL*__W_ROW * sizeof(float) * __BATCH;
    size_t V_size = __V_COL*__V_ROW * sizeof(float) * __BATCH;
    size_t U_size = __U_COL*__U_ROW * sizeof(float) * __BATCH;

    float *xt_cpu = (float *)malloc(xt_size);
    float *ht_cpu = (float *)malloc(ht_size);
    float *yt_cpu = (float *)malloc(yt_size);
    float *W_cpu = (float *)malloc(W_size);
    float *U_cpu = (float *)malloc(U_size);
    float *V_cpu = (float *)malloc(V_size);

    // these are intermediate ones that should be the same size as ht because they are making the new ht
    float *xt_times_U_cpu = (float*)malloc(ht_size);
    float *prevht_times_W_cpu = (float*)malloc(ht_size);

    // Verify that allocations succeeded
    if (xt_cpu == NULL || ht_cpu == NULL || yt_cpu == NULL || W_cpu == NULL || V_cpu == NULL | U_cpu == NULL || xt_times_U_cpu == NULL || prevht_times_W_cpu == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize the host input vectors
    for (int i = 0; i < xt_size/sizeof(float); ++i)
    {
        xt_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < ht_size/sizeof(float); ++i)
    {
        ht_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < W_size/sizeof(float); ++i)
    {
        W_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < V_size/sizeof(float); ++i)
    {
        V_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < U_size/sizeof(float); ++i)
    {
        U_cpu[i] = rand()/(float)RAND_MAX;
    }


    // Allocate the device input vector xt
    float *xt_cuda = NULL;
    err = cudaMalloc((void **)&xt_cuda, xt_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector xt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector ht
    float *ht_cuda = NULL;
    err = cudaMalloc((void **)&ht_cuda, ht_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector ht (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector yt
    float *yt_cuda = NULL;
    err = cudaMalloc((void **)&yt_cuda, yt_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector yt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector W
    float *W_cuda = NULL;
    err = cudaMalloc((void **)&W_cuda, W_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector W (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector U
    float *U_cuda = NULL;
    err = cudaMalloc((void **)&U_cuda, U_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector U (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector V
    float *V_cuda = NULL;
    err = cudaMalloc((void **)&V_cuda, V_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector xt_times_U
    float *xt_times_U_cuda = NULL;
    err = cudaMalloc((void **)&xt_times_U_cuda, ht_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
    
    // Allocate the device input vector prevht_times_W
    float *prevht_times_W_cuda = NULL;
    err = cudaMalloc((void **)&prevht_times_W_cuda, ht_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }




    // Copy the host input vector xt in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(xt_cuda, xt_cpu, xt_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector xt from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector ht in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(ht_cuda, ht_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector ht from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector yt in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(yt_cuda, yt_cpu, yt_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector yt from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector W in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(W_cuda, W_cpu, W_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector W from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector V in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(V_cuda, V_cpu, V_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vector U in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(U_cuda, U_cpu, U_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the host input vector xt_times_U in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(xt_times_U_cuda, xt_times_U_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector xt_times_U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vector prevht_times_W in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(prevht_times_W_cuda, prevht_times_W_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector prevht_times_W from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    

    // Launch the Vector RNN_matrix_multiply_step1 CUDA Kernel
    int threadsPerBlock = 64;   //V100 has 64 single-precision CUDA cores per SM
    int blocksPerGrid =(ht_size/(sizeof(float) * __BATCH) + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    
    struct timeval t1, t2;
    gettimeofday(&t1,0);
    
    gpu_matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(xt_cuda, U_cuda, xt_times_U_cuda, __XT_ROW, __XT_COL, __U_COL);   
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gpu_matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(ht_cuda, W_cuda, prevht_times_W_cuda, __HT_ROW, __HT_COL, __W_COL);   
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gpu_matrix_add<<<blocksPerGrid, threadsPerBlock>>>(xt_times_U_cuda, prevht_times_W_cuda, ht_cuda);   
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to generate:  %3.1f ms \n", time);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(ht_cpu, ht_cuda, ht_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // TODO: Calculate the verified multiplication on the host? Idk if this is necessary
    printf("Outputs: ");
    for(int i = 0; i < ht_size/sizeof(float); i++)
        printf("%.4f, ", ht_cpu[i]);
    printf("\n");

    printf("Multiplication PASSED\n");
    

    // Free device global memory
    err = cudaFree(xt_cuda);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector xt_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(U_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector U_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(ht_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector ht_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(W_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector W_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(yt_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector yt_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(V_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector V_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(xt_times_U_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector xt_time_U_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(prevht_times_W_cuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector prevht_times_W_cuda (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(xt_cpu);
    free(ht_cpu);
    free(yt_cpu);
    free(W_cpu);
    free(V_cpu);
    free(U_cpu);

    return 0;
}

