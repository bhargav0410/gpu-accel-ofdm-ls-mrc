#ifndef _SHMEMSYMBUFF_HPP_
#define _SHMEMSYMBUFF_HPP_

#include <iostream>
#include<math.h>
#include<fstream>
#include <cstdlib>
#include <cstring>
#include <complex>
#include "CSharedMemSimple.hpp"

//Timer
#include<time.h>

//16 x 1024
#ifndef numOfRows
	#define numOfRows 16
#endif

#ifndef dimension
	#define dimension 1024
#endif
//64
#ifndef prefix
  #define prefix 0
#endif

#ifndef timerEnabled
  #define timerEnabled true
#endif

#ifndef testEnabled
  #define testEnabled true
#endif

std::ofstream outfile;

//100
#ifndef lenOfBuffer
	#define lenOfBuffer 10
#endif
#define numberOfSymbolsToTest lenOfBuffer
#define shmemID "/blah"
#define PL printf("Line #: %d \n", __LINE__);
#define timerEn timerEnabled
#define testEn testEnabled

//Number of times the program is to be run
int numTimes = 1;

//Timing
float readT[numberOfSymbolsToTest];
//First one is time to solve for H
float decode[numberOfSymbolsToTest];
//Time to drop prefix
float drop[numberOfSymbolsToTest];
float fft[numberOfSymbolsToTest];

//complexNumber
struct complexF{
	float real;
	float imag;
};

//Symbol-> one matrix 16x1088 of complex numbers
struct symbol{
	complexF data[numOfRows*(dimension+prefix)];
};

/* Defines "structure" of shared memory */
struct symbolBuffer {        
  //amount of symbols in the buffer
  int size;
  //Will always 
  int readPtr;
  //Initialized to -1
  int writePtr;
  //Buffer of Symbols
  symbol symbols[lenOfBuffer];
};

void printOutArr(complexF* a, int rows, int cols){
	for (int i = 0; i<rows; i++){  
		for (int j = 0; j<cols; j++){
			std::cout<<"("<<a[i*cols + j].real<<", "<<a[i*cols + j].imag<<"), ";
		}
		printf("\n");
	}
}

void printInfo(){
	printf("\tSymbol Dimension(w/o prefix) = %d x %d \n", numOfRows, dimension);
	printf("\tPrefix = %d\n", prefix);
	printf("\t# Of Symbols To Test = %d\n", numberOfSymbolsToTest);
	
}

//real=avg and imag = var
complexF findAvgAndVar(float* times, int amt){
	float avgTime = 0;
	
	for(int i =0; i< amt; i++){
		//print out
		//printf("Time = %e \n", times[i]);
		avgTime = avgTime+ times[i];
	}
	//Find avg 
	avgTime = avgTime/amt;
	
	float variance = 0;
	for(int i=0; i<amt; i++){
		variance = variance + (times[i]-avgTime) *(times[i]-avgTime);
	}
	variance = variance/amt;
	
	complexF c;
	c.real = avgTime;
	c.imag= variance;
	return c;
			
}

void printTimes(bool cpu){
	complexF readtime = findAvgAndVar(readT, numberOfSymbolsToTest);
	complexF decodetime = findAvgAndVar(&decode[1], numberOfSymbolsToTest-1);
	complexF FFTtime = findAvgAndVar(fft, numberOfSymbolsToTest);
	printf("\t \t Avg Time(s) \t Variance (s^2) \n");
	printf("Read: \t \t %e \t %e \n", readtime.real/numTimes, readtime.imag/numTimes);
	printf("ChanEst: \t %e \n", decode[0]/numTimes);
	printf("Decode: \t %e \t %e \n", decodetime.real/numTimes, decodetime.imag/numTimes);
	printf("FFT: \t \t %e \t %e \n", FFTtime.real/numTimes, FFTtime.imag/numTimes);
	
	if(cpu){
		complexF dropTime = findAvgAndVar(drop, numberOfSymbolsToTest);
		printf("Drop: \t \t %e \t %e \n", dropTime.real/numTimes, dropTime.imag/numTimes);
		
	}
}

void storeTimes(bool cpu) {
	complexF readtime = findAvgAndVar(readT, numberOfSymbolsToTest);
	complexF decodetime = findAvgAndVar(&decode[1], numberOfSymbolsToTest-1);
	complexF FFTtime = findAvgAndVar(fft, numberOfSymbolsToTest);
	complexF dropTime = findAvgAndVar(drop, numberOfSymbolsToTest);
	readtime.real = readtime.real/numTimes;
	decodetime.real = decodetime.real/numTimes;
	FFTtime.real = FFTtime.real/numTimes;
	dropTime.real = dropTime.real/numTimes;
	decode[0] = decode[0]/numTimes;
	std::string file;
	if (cpu == false){
		file = "time_gpu.dat";
	}
	else {
		file = "time_cpu.dat";
	}
	outfile.open(file.c_str(), std::ofstream::binary);
	outfile.write((const char*)&readtime.real, sizeof(float));
	outfile.write((const char*)&decode[0], sizeof(float));
	outfile.write((const char*)&decodetime.real, sizeof(float));
	outfile.write((const char*)&FFTtime.real, sizeof(float));
	outfile.write((const char*)&dropTime.real, sizeof(float));
}

int buffIter = 0;

class ShMemSymBuff{
	private:
		symbolBuffer* buff;
		CSharedMemSimple* shm_bufPtr;
		bool isMast;
		clock_t start, finish;

	public:
		// Constructor - create a shared memory space for symbols
		ShMemSymBuff(std::string shm_uid, int isMaster){
			shm_bufPtr = new CSharedMemSimple(shm_uid, sizeof(struct symbolBuffer));
			buff = (struct symbolBuffer *)shm_bufPtr->ptr();
			isMast=false;
			if(isMaster==1){
				isMast=true;
				shm_bufPtr->set_master_mode();
				buff->size = lenOfBuffer;
				buff->writePtr=-1;
				buff->readPtr=0;
			}
			else{
				//wait for master to create it
				while(buff->size<=0);
			}
			//shm_bufPtr->info();	
		}

		// Destructor - releases shared memory
		~ShMemSymBuff(){
			if(isMast){
				while(buff->size == -1){
					delete shm_bufPtr;
				}
			}
			else{
				buff->size = -1;
			}
		}

		void info(){
			shm_bufPtr->info();
		}
		
		//Reads a whole symbol into Y -> use prefix definition to determine prefix
		void readNextSymbol(complexF* Y, int it){
			int rows = numOfRows;
			int cols = dimension;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);
			
			complexF* temp = 0;
			temp=(complexF*)malloc(rows*(cols+prefix)* sizeof(*temp));

			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			//time start
			if(timerEn){
				start = clock();
			}
			if(prefix>0){ //Read with prefix 
				int size= rows*(cols+prefix)*sizeof(*temp);
				// read data from shared mem into temp
				memcpy(temp,&buff->symbols[buff->readPtr].data[0], size);
			}
			else{ //Read without prefix
				int size= rows*(cols)* sizeof (*Y);
				// read data from shared mem into Y
				memcpy(Y,&buff->symbols[buff->readPtr].data[0], size);
			}
			if(timerEn){
				finish = clock();
				readT[it] = readT[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
			}
			
			//once you are done reading to that spot
			int p = buff->readPtr+1;
			//can't read until writer writes so don't change it yet
			while(buff->writePtr == p );
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				while(buff->writePtr == 0 );
				buff->readPtr=0;
			} 
			else{
				buff->readPtr=p;
			}
			
			if(prefix>0){ //Drop the prefix
				//time start
				if(timerEn){
					start = clock();
				}
				//drop the 64 element prefix
				for(int i=0; i<rows; i++){
					memcpy(&Y[i*cols], &temp[i*(cols+prefix)+ prefix], cols*sizeof(*Y));
				}
				if(timerEn){
					finish = clock();
					drop[it] = drop[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
				}
			}
			
			free(temp);
		}
		
		//Reads a last symbol into Y doesn't worry about changing ptr index to same as writer since last one
		void readLastSymbol(complexF* Y){
			int rows = numOfRows;
			int cols = dimension;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);

			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			
			complexF* temp = 0;
			temp=(complexF*)malloc(rows*(cols+prefix)* sizeof (*temp));
			
			
			// read data from shared mem into temp
			memcpy(temp,&buff->symbols[buff->readPtr].data[0], rows*(cols+prefix)* sizeof (*temp));
			//once you are done reading to that spot
			int p = buff->readPtr+1;
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				buff->readPtr=0;
			} 
			else{
				buff->readPtr=p;
			}
			
			//drop the 64 element prefix
			for(int i=0; i<rows; i++){
				memcpy(&Y[i*cols], &temp[i*(cols+prefix)+ prefix], cols*sizeof(*Y));
			}
			free(temp);
		}
		
		#ifdef cudaEn
		//Read symbol into device memory with prefix
		void readNextSymbolCUDA(complexF *dY, int it){
			int rows = numOfRows;
			int cols = dimension+prefix;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);
			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			//time start
			if(timerEn){
				start = clock();
			}
			int size = rows*(cols)* sizeof (*dY);
			/*
			complexF* Y = 0;
			Y = (complexF*)malloc(size);
			memcpy(Y,&buff->symbols[buff->readPtr].data[0], size);
			*/
			// read data from shared mem into Y
		//	if (it == 1) {
			std::string file = "Sym_copy_sh_mem.dat";
		//	cuFloatComplex Yf[rows*cols];
			complexF *Yf;
			Yf = (complexF*)malloc(rows*cols*sizeof(*Yf));
			memcpy(Yf,&buff->symbols[buff->readPtr].data[0], size);
			outfile.open(file.c_str(), std::ofstream::binary);
			outfile.write((const char*)Yf, rows*(cols)*sizeof(*Yf));
			outfile.close();
		//	}
			cudaMemcpy(dY, &buff->symbols[buff->readPtr].data[0], size, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			
			/*
			for (int i = 0; i < cols; i++) {
				printf("( %f, %f),",dY[i].x,dY[i].y);
			}
			std::cout << std::endl;
			*/
			
			if(timerEn){
				finish = clock();
				readT[it] = readT[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
			}
			//once you are done reading to that spot
			int p = buff->readPtr+1;
			//can't read until writer writes so don't change it yet
			while(buff->writePtr == p );
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				while(buff->writePtr == 0 );
				buff->readPtr=0;
			} 
			else{
				buff->readPtr=p;
			}
		}
			
		//Reads a last symbol into Y doesn't worry about changing ptr index to same as writer since last one
		void readLastSymbolCUDA(complexF* dY){
			int rows = numOfRows;
			int cols = dimension+prefix;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);

			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			
			int size = rows*(cols)* sizeof (*dY);
			/*
			complexF* Y = 0;
			Y = (complexF*)malloc(size);
			memcpy(Y,&buff->symbols[buff->readPtr].data[0], size);
			
			// read data from shared mem into Y
			cudaMemcpy(dY, Y, size, cudaMemcpyHostToDevice);
			*/
			// read data from shared mem into Y
			cudaMemcpy(dY, &buff->symbols[buff->readPtr].data[0], size, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			//once you are done reading to that spot
			int p = buff->readPtr+1;
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				buff->readPtr=0;
			} 
			else{
				buff->readPtr=p;
			}
		}
		
		#endif
		
		//Writes a Symbol into this buffer
		void writeNextSymbolWithWait(std::complex<float>* Yf){
			int rows = numOfRows;
			int cols = dimension+prefix;
			//writePtr==-1 to start
			if(buff->writePtr ==-1){
				//you can write
				memcpy(&buff->symbols[0].data[0], Yf, rows*cols* sizeof (*Yf));
				buff->writePtr=1;
				return;
			}
			//if you are trying to write to the index that reader is reading
			while(buff->writePtr == buff->readPtr);
			
			memcpy(&buff->symbols[buff->writePtr].data[0], Yf, rows*cols* sizeof (*Yf));
			
			//once you are done writing to that spot
			int p = buff->writePtr+1;
			//can't read until writer writes so don't change it yet
			while(buff->readPtr == p );
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				//can't read until writer writes so don't change it yet
				while(buff->readPtr == 0 );
				buff->writePtr=0;
			} 
			else{
				buff->writePtr=p;
			}
			
		}

		void writeNextSymbolNoWait(std::complex<float>* Yf){
			int rows = numOfRows;
			int cols = dimension+prefix;
			//writePtr==-1 to start
			if(buff->writePtr ==-1){
				//you can write
				memcpy(&buff->symbols[0].data[0], Yf, rows*cols* sizeof (*Yf));
				buff->writePtr=1;
				return;
			}
			memcpy(&buff->symbols[buff->writePtr].data[0], Yf, rows*cols* sizeof (*Yf));
			
			//once you are done writing to that spot
			int p = buff->writePtr+1;
			
			//if you reach the end of the buffer
			if(p >= lenOfBuffer){
				buff->writePtr=0;
			} 
			else{
				buff->writePtr=p;
			}
		}
		
};


#endif
