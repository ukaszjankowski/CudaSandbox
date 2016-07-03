#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char gray;
};

__declspec(dllexport) cudaError_t Grayscale(pixel *data, unsigned int size);

cudaError_t CudaFail(const char *message, pixel *data, cudaError_t status);

#endif