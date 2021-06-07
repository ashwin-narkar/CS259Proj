#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>

#define __CLASS 1
#define __BATCH 1

#if __CLASS == 1
    #define __INPUTLAYER 10
    #define __OUTLAYER 7
#else
    #define __INPUTLAYER 4096
    #define __OUTLAYER 1024
#endif


__global__ void 
mv_multiply( float* C, float* A, float* B, int n, int m)
{
    int gindex = threadIdx.x + blockIdx.x*blockDim.x;

    C[gindex] = 0.0f;

    for (int b = 0; b < __BATCH; b++)
    {
        for (int k = 0; k < n; k++){
            C[gindex+b*m] += A[k+n*b] * B[gindex*n + k];
        }
    }

    
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t xt_size = __INPUTLAYER * sizeof(float) * __BATCH;
    size_t ht_size = __INPUTLAYER * __OUTLAYER * sizeof(float);
    size_t yt_size = __OUTLAYER * sizeof(float) * __BATCH;
    size_t W_size = __OUTLAYER * __OUTLAYER * sizeof(float);
    size_t V_size = __INPUTLAYER * __BATCH * sizeof(float);
    size_t U_size = __OUTLAYER * __BATCH * sizeof(float);

    // Allocate the host input vector xt
    float *xt_cpu = (float *)malloc(inputSize);

    // Allocate the host input vector ht
    float *ht_cpu = (float *)malloc(matrixSize);

    // Allocate the host output vector yt
    float *yt_cpu = (float *)malloc(outSize);

    float *W_cpu = (float *)malloc(W_size);

    float *U_cpu = (float *)malloc(U_size);
    
    float *V_cpu = (float *)malloc(V_size);

    // these are intermediate ones that should be the same size as ht because they are making the new ht
    float *xt_times_U_cpu = (float*)malloc(ht_size);
    float *prevht_times_W_cpu = (float*)malloc(ht_size);


    float prevht_times_W[70];

    // Verify that allocations succeeded
    if (xt_cpu == NULL || ht_cpu == NULL || yt_cpu == NULL || W_cpu == NULL || V_cpu == NULL | U_cpu == NULL || xt_times_U_cpu == NULL || prevht_times_W_cpu == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize the host input vectors
    for (int i = 0; i < xt_size; ++i)
    {
        xt_cpu[i] = rand()/(float)RAND_MAX;
    }

    for (int i = 0; i < ht_size; ++i)
    {
        ht_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < W_size; ++i)
    {
        W_cpu[i] = rand()/(float)RAND_MAX;
    }

    for (int i = 0; i < V_size; ++i)
    {
        V_cpu[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < U_size; ++i)
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




    // Copy the host input vector xt in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(xt_cuda, xt_cpu, xt_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector xt from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector ht in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(ht_cuda, ht_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector ht from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector yt in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(yt_cuda, yt_cpu, yt_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector yt from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector W in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(W_cuda, W_cpu, W_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector W from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector V in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(V_cuda, V_cpu, V_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vector U in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(U_cuda, U_cpu, U_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the host input vector xt_times_U in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(xt_times_U_cuda, xt_times_U_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector xt_times_U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vector prevht_times_W in host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(prevht_times_W_cuda, prevht_times_W_cpu, ht_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector prevht_times_W from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    

    // Launch the Vector RNN_matrix_multiply_step1 CUDA Kernel
    int threadsPerBlock = 64;   //V100 has 64 single-precision CUDA cores per SM
    int blocksPerGrid =(__OUTLAYER + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


    struct timeval t1, t2;
    gettimeofday(&t1,0);
    mv_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, __INPUTLAYER, __OUTLAYER);   
    cudaThreadSynchronize();
    err = cudaGetLastError();

    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to generate:  %3.1f ms \n", time);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, outSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // TODO: Calculate the verified multiplication on the host? Idk if this is necessary

    printf("Multiplication PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
