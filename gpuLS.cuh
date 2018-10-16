/*
Copyright (c) 2018, WINLAB, Rutgers University, USA
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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