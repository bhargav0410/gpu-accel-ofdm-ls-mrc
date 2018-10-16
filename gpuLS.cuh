#ifndef _GPULS_CUH_
#define _GPULS_CUH_

#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_gpu.hpp"
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define FFT_size dimension
#define cp_size prefix
#define numSymbols lenOfBuffer

//gpu

#define threadsPerBlock FFT_size
#define numOfBlocks numOfRows

//LS
#define fileNameForX "Pilots.dat"
#define mode 0
/*
	mode:
		= 1 -> master -> creates shared memory 
		= 0 -> slave -> doesn't create the shared memory
*/
 
//!How to Compile:   nvcc ../../examples/gpuLS_cucomplex.cu -lcufft -lrt -o gpu -arch=sm_35
// ./gpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023

class gpuLS {
	public:
		ShMemSymBuff* buffPtr;
		cudaDeviceProp devProp;

		gpuLS();
		~gpuLS();
		
		//Reads in Vector X from file -> 1xcols
		void matrix_readX(cuFloatComplex*, int);

		void copyPilotToGPU(cuFloatComplex*, int, int);
		
		void shiftOneRowCPU(cuFloatComplex*, int, int);
		
		void ShiftOneRow(cuFloatComplex*, int, int, dim3, dim3, cudaStream_t*);
		
		void DropPrefix(cuFloatComplex*, cuFloatComplex*, int, int, dim3, dim3, cudaStream_t*);
		
		void FindLeastSquaresGPU(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, int, int, dim3, dim3, cudaStream_t*);
		
		void FindHsqrdforMRC(cuFloatComplex*, float*, int, int, dim3, dim3, cudaStream_t*);
		
		void MultiplyWithChannelConj(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, int, int, int, dim3, dim3, cudaStream_t*);
		
		void CombineForMRC(cuFloatComplex*, float*, int, int, dim3, dim3, cudaStream_t*);
		
		void batchedFFT(cuFloatComplex*, int, int, cudaStream_t*);

		void firstVector(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int, int);

		void demodOneSymbol(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int, int);

		void demodOneFrame(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int);

		void demodOneFrameCUDA(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int);

		void demodOptimized(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int);

		void demodCuBlas(cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, int, int);

};
#endif