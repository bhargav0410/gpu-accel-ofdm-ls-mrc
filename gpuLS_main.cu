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

#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_cucomplex.hpp"
#include "gpuLS.cuh"
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <assert.h>
#define FFT_size dimension
#define cp_size prefix
#define numSymbols lenOfBuffer


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

using namespace std;

std::string file = "Output_gpu.dat";
//std::ofstream outfile;

int main(){
	int rows = numOfRows; // number of vectors
	int cols=dimension;//dimension
	cudaSetDevice(0);
	//printf("CUDA LS: \n");
	//printInfo();
	//dY holds symbol with prefix
	cuFloatComplex *dY = 0;
	dY = (cuFloatComplex*)malloc(rows*(cols)* sizeof (*dY));
	
	float *Hsqrd = 0;
	cudaMalloc((void**)&Hsqrd, (cols-1)* sizeof (*Hsqrd));
	
	//dH (and Hconj) = 16x1023
	cuFloatComplex *dH = 0;
	cudaMalloc((void**)&dH, rows*(cols-1)* sizeof (*dH));
	
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex *dX = 0;
	cudaMalloc((void**)&dX, rows*(cols-1)* sizeof (*dX));
	
	cuFloatComplex *Yf = 0;
	Yf = (cuFloatComplex*)malloc((cols-1)* sizeof (*Yf));
	
	cuFloatComplex* Y = 0;
	cudaMalloc((void**)&Y, rows*cols*sizeof(*Y));
	
	
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	
	
	//Shared Memory
	//string shm_uid = shmemID;
	//gpu->buffPtr = new ShMemSymBuff(shm_uid, mode);
	
	copyPilotToGPU(dX, rows, cols);
	
	for (int iter = 0; iter < numTimes; iter++) {
	firstVector(dY, Y, dH, dX, Hsqrd, rows, cols, 0);
	//dH holds h conj
	//dX holds |H|^2
	for(int i=1; i<numberOfSymbolsToTest; i++){
		
		demodOneSymbol(dY, Y, dH, Hsqrd, rows, cols, i);
		
		if(testEn){
			//printf("Symbol #%d:\n", i);
			//cuda copy it over
			memcpy(Yf, dY, (cols-1)* sizeof (*Yf));
			if (i <= 1) {
				outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
			} else {
				outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::app);
			}
			outfile.write((const char*)Yf, (cols-1)*sizeof(*Yf));
			outfile.close();
			//printOutArr(Yf, 1, cols-1);
		}
		
		
	}
	}
	free(Yf);
	free(dY);
	cudaFree(Y);
	cudaFree(dH);
	cudaFree(dX);
	cudaFree(Hsqrd);
	//delete buffPtr;
	
	if(timerEn) {
		printTimes(true);
		storeTimes(false);
	}
	return 0;

}