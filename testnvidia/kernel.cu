// Image processing in NVIDIA CUDA
// Copyright 2016 by £ukasz Jankowski

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// RGB To Greenscale conversion factors
#define RED_FACTOR		0.299
#define GREEN_FACTOR	0.587
#define BLUE_FACTOR		0.114

__global__ void GrayscaleKernel(pixel *data)
{
    int i = threadIdx.x;
	data[i].gray = RED_FACTOR   * data[i].r 
				 + GREEN_FACTOR * data[i].g 
				 + BLUE_FACTOR  * data[i].b;
}

__global__ void InvertKernel(pixel *data)
{
	int i = threadIdx.x;
	data[i].r = 255 - data[i].r;
	data[i].g = 255 - data[i].g;
	data[i].b = 255 - data[i].b;
}

__global__ void ContrastKernel(pixel *data, float contrast)
{
	if (contrast < 0) contrast = 0;
	if (contrast > 2) contrast = 2;

	int i = threadIdx.x;
	data[i].r = contrast * data[i].r < data[i].r ? 255 : contrast * data[i].r;
	data[i].g = contrast * data[i].g < data[i].g ? 255 : contrast * data[i].g;
	data[i].b = contrast * data[i].b < data[i].b ? 255 : contrast * data[i].b;
}

int main()
{
    const int arraySize = 1024;
	pixel *data = (pixel*)malloc(arraySize * sizeof(pixel));

	cudaError_t cudaStatus;

	for (long i = 0; i < arraySize; i++) {
		data[i].gray = 0;
		data[i].r = 10;
		data[i].g = 20;
		data[i].b = 30;
	}

	time_t cudaStartTime = time(NULL);
	cudaStatus = Grayscale(data, arraySize);
	time_t cudaEndTime = time(NULL);
	printf("GPU: %i\n", cudaEndTime - cudaStartTime);

	time_t cpuStartTime = time(NULL);
	for (size_t i = 0; i < 1000000; i++) {
		for (size_t j = 0; j < arraySize; j++)
		{
			data[j].gray = RED_FACTOR   * data[j].r
						 + GREEN_FACTOR * data[j].g
						 + BLUE_FACTOR  * data[j].b;
		}
	}
	time_t cpuEndTime = time(NULL);
	printf("CPU (single thread): %i\n", cpuEndTime - cpuStartTime);

	//printf("Pixel: %d\n", data[0].gray);

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda failed!");
		getchar();
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		getchar();
        return 1;
    }

	getchar();
    return 0;
}

__declspec(dllexport) cudaError_t Grayscale(pixel *data, unsigned int size)
{
	pixel *dev_data = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return CudaFail("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", data, cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(pixel));
	if (cudaStatus != cudaSuccess) return CudaFail("cudaMalloc failed!\n", data, cudaStatus);

	cudaStatus = cudaMemcpy(dev_data, data, size * sizeof(pixel), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return CudaFail("cudaMemcpy failed!\n", data, cudaStatus);

    // Launch a kernel on the GPU with one thread for each element.
	for (int i = 0; i < 1000000; i++) {
		GrayscaleKernel<<<1, size >>>(dev_data);
	}

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) return CudaFail("Kernel launch failed\n", data, cudaStatus);
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) return CudaFail("cudaDeviceSynchronize returned error after launching Kernel!\n", data, cudaStatus);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(data, dev_data, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return CudaFail("cudaMemcpy failed!\n", data, cudaStatus);

    cudaFree(dev_data);
    return cudaStatus;
}

cudaError_t CudaFail(const char *message, pixel *data, cudaError_t status) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", status);
	cudaFree(data);
	return status;
}