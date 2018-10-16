#ifndef _SHMEMSYMBUFF_HPP_
#define _SHMEMSYMBUFF_HPP_

#include <iostream>
#include<math.h>
#include<fstream>
#include <cstdlib>
#include <cstring>
#ifdef cudaEn
#include <cuComplex.h>
#endif
#include <complex>
//#include <boost/program_options.hpp>
//#include <boost/thread.hpp>
#include "CSharedMemSimple.hpp"

//Timer
#include<time.h>



//16 x 1024
#ifndef numOfRows
	#define numOfRows 16
#endif

#ifndef numUsers
	#define numUsers 4
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

#ifndef lenOfBuffer
	#define lenOfBuffer 101
#endif
#define numberOfSymbolsToTest lenOfBuffer
#define shmemID "/blah"
#define PL printf("Line #: %d \n", __LINE__);
#define timerEn timerEnabled
#define testEn testEnabled

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


class ShMemSymBuff{
	private:
		symbolBuffer* buff;
		CSharedMemSimple* shm_bufPtr;
		bool isMast;
		clock_t start, finish;
		//Timing
		float readT[numberOfSymbolsToTest];
		//First one is time to solve for H
		float decode[numberOfSymbolsToTest];
		//Time to drop prefix
		float drop[numberOfSymbolsToTest];
		float fft[numberOfSymbolsToTest];
		cudaStream_t streams[lenOfBuffer];
		
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
		
		void setReadT(float value, int iter) {
			readT[iter] = value;
		}
		
		void setFft(float value, int iter) {
			fft[iter] = value;
		}
		
		void setDecode(float value, int iter) {
			decode[iter] = value;
		}
		
		void setDrop(float value, int iter) {
			drop[iter] = value;
		}
		
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
			c.imag = variance;
			return c;
					
		}

		void printTimes(bool cpu){
			complexF readtime = findAvgAndVar(readT, numberOfSymbolsToTest);
			complexF decodetime = findAvgAndVar(&decode[1], numberOfSymbolsToTest-1);
			complexF FFTtime = findAvgAndVar(&fft[1], numberOfSymbolsToTest-1);
			printf("\t \t Avg Time(s) \t Variance (s^2) \n");
			printf("R/W: \t \t %e \t %e \n", readtime.real, readtime.imag);
			printf("ChanEst: \t %e \n", (decode[0] + FFTtime.real + readtime.real));
			printf("Mod/Demod: \t %e \t %e \n", decodetime.real, decodetime.imag);
			printf("FFT: \t \t %e \t %e \n", FFTtime.real, FFTtime.imag);
			printf("Frame: \t \t %e \n", (((FFTtime.real + readtime.real + decodetime.real)*(lenOfBuffer-1))));
			
			if(cpu){
				complexF dropTime = findAvgAndVar(drop, numberOfSymbolsToTest);
				printf("Drop: \t \t %e \t %e \n", dropTime.real, dropTime.imag);
				
			}
		}

		void storeTimes(bool cpu) {
			std::ofstream outfile;
			complexF readtime = findAvgAndVar(readT, numberOfSymbolsToTest);
			complexF decodetime = findAvgAndVar(&decode[1], numberOfSymbolsToTest-1);
			complexF FFTtime = findAvgAndVar(fft, numberOfSymbolsToTest-1);
			complexF dropTime = findAvgAndVar(drop, numberOfSymbolsToTest);
			readtime.real = readtime.real;
			decodetime.real = decodetime.real;
			FFTtime.real = FFTtime.real;
			dropTime.real = dropTime.real;
			decode[0] = decode[0];
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
			outfile.close();
		}
		
		void setBuffLen(int size_) {
			buff->size = size_;
		}
		
		//Reads a whole symbol into Y -> use prefix definition to determine prefix
		template <typename T>
		void readNextSymbol(T* Y, int it){
			int rows = numOfRows;
			int cols = dimension;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);
			
			T* temp = 0;
			temp=(T*)malloc(rows*(cols+prefix)* sizeof(*temp));

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
				readT[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
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
					drop[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
				}
			}
			
			free(temp);
		}
		
		//Reads a last symbol into Y doesn't worry about changing ptr index to same as writer since last one
		template <typename T>
		inline void readLastSymbol(T* Y){
			int rows = numOfRows;
			int cols = dimension;
			
			//writePtr==-1 to start
			while(buff->writePtr ==-1);

			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			
			T* temp = 0;
			temp=(T*)malloc(rows*(cols+prefix)* sizeof (*temp));
			
			
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
		
		cudaStream_t * createStream(int it) {
			cudaStreamCreate(&streams[it]);
			return &streams[it];
		}
		
		void destroyStream(int it) {
			cudaStreamDestroy(stream[it]);
		}
		
		//Read symbol into device memory with prefix
		template <typename T>
		inline void readNextSymbolCUDA(T* dY, int it){
			int rows = numOfRows;
			int cols = dimension;
			//writePtr==-1 to start
			while(buff->writePtr ==-1);
			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			//time start
			if(timerEn){
				start = clock();
			}
			int size = rows*(cols+prefix)* sizeof (*dY);
			cudaMemcpyAsync(dY, &buff->symbols[buff->readPtr].data[0], size, cudaMemcpyHostToDevice, streams[it]);
				
			
			if(timerEn){
				finish = clock();
				readT[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
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
		template <typename T>
		inline void readLastSymbolCUDA(T* dY){
			int rows = numOfRows;
			int cols = dimension;
			cudaStreamCreate(&streams[lenOfBuffer-1]);
			//writePtr==-1 to start
			while(buff->writePtr ==-1);

			//can't read until writer writes
			while(buff->writePtr == buff->readPtr );
			
			if(timerEn){
				start = clock();
			}
			int size = rows*(cols+prefix)* sizeof (*dY);
			// read data from shared mem into Y
			cudaMemcpyAsync(dY,&buff->symbols[buff->readPtr].data[0], size, cudaMemcpyHostToDevice, streams[lenOfBuffer-1]);
			
			
			if(timerEn){
				finish = clock();
				readT[numberOfSymbolsToTest-1] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
			}
			
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
		template <typename T>
		inline void writeNextSymbolWithWait(T* Yf){
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
				while(buff->readPtr >= 0 );
				buff->writePtr=0;
			} 
			else{
				buff->writePtr=p;
			}
			
		}

		template <typename T>
		inline void writeNextSymbolNoWait(T* Yf){
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
