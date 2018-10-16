#ifndef _CPULS_HPP_
#define _CPULS_HPP_

//FFTW library 
#include <fftw3.h>
//Shared Memory 
#include "CSharedMemSimple.hpp"
#include "ShMemSymBuff_cucomplex.hpp"
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <cblas.h>
#define fileNameForX "Pilots.dat"

/*
	mode:
		= 1 -> master -> creates shared memory 
		= 0 -> slave -> doesn't create the shared memory
		
	Waits to read dimension vector then does fft on it and then divides by 1+i 
*/

//! Install dependencies: apt-get -y install libboost-program-options-dev libfftw3-dev 
//!How to Compile:   g++ -o cpu ../../examples/cpuLS.cpp -lfftw3f -lrt
// ./cpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023

using namespace std;
ShMemSymBuff* buffPtr;
#define mode 1
string file = "Output_cpu.dat";
string in_file = "Input_cpu.dat";

int num_syms;

//LAPACK and BLAS functions

extern "C" {
	void cgetrf_(int* m, int* n, complexF* A, int* lda, int* ipiv, int* info);
	void cgetri_(int* n, complexF* A, int* lda, int* ipiv, complexF* work, int* lwork, int* info);
	void csytrf_(char* uplo, int* n, complexF* A, int* lda, int* ipiv, complexF* work, int* lwork, int* info);
	void csytri_(char* uplo, int* n, complexF* A, int* lda, int* ipiv, complexF* work, int* info);
	float clange_(char* norm, int* m, int* n, complexF* A, int* lda, float* work);
}


//Reads in Vector X from file -> 1xcols
void matrix_readX(complexF* X, int cols){
	ifstream inFile;
	inFile.open(fileNameForX, std::ifstream::binary);
	if (!inFile) {
		cerr << "Unable to open file data file, filling in 1+i for x\n";
		float c=0.707f;
		for (int col = 0; col <  cols; col++){
			X[col].real=c;
			X[col].imag=c;
		}
		return;
	}
	
	inFile.read((char*)X, (cols)*sizeof(*X));
	//printOutArr(X, 1, cols);
	/*
	float c=0;
	for (int col = 0; col <  cols; col++){
		inFile >> c;
		X[col].real=c;
		inFile >> c;
		X[col].imag=c;
	}
	*/
	
	complexF* temp = 0;
	temp=(complexF*)malloc ((cols-1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &X[(cols+1)/2], (cols-1)/2* sizeof (*X));
	//copy first half to second half
	memmove(&X[(cols-1)/2], X, (cols+1)/2* sizeof (*X));
	//copy temp to first half
	memmove(X, temp, (cols-1)/2* sizeof (*X));
	
	free(temp);
	
	inFile.close();
}

void ifftShiftOneRow(complexF* Y, int cols, int row){
	complexF* Yf = &Y[row*cols];
	
	complexF* temp = 0;
	temp=(complexF*)malloc ((cols)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &Yf[(cols)/2], (cols)/2* sizeof (*Yf));
	//copy first half to second half
	memmove(&Yf[(cols)/2], Yf, (cols)/2* sizeof (*Yf));
	//copy temp to first half
	memmove(Yf, temp, (cols)/2* sizeof (*Yf));
	
	free(temp);
}

//Shifts first and second half of vector fft(Y)
void shiftOneRow(complexF* Y, int cols, int row){
	complexF* Yf = &Y[row*cols];
	//std::cout << "Here...\n";
	complexF* temp = 0;
	temp=(complexF*)malloc ((cols+1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &Yf[(cols-1)/2], (cols+1)/2* sizeof (*Yf));
	//copy first half to second half
	memmove(&Yf[(cols+1)/2], Yf, (cols-1)/2* sizeof (*Yf));
	//copy temp to first half
	memmove(Yf, temp, (cols+1)/2* sizeof (*Yf));
	
	free(temp);
	
}

//IFFT on one vector of Y in row
void ifftOneRow(complexF* Y, int cols, int row){
	
	fftwf_complex* org = (fftwf_complex*)&Y[row*cols];
	fftwf_complex* after = (fftwf_complex*)&Y[row*cols];

	fftwf_plan fft_p = fftwf_plan_dft_1d(cols, org, after, FFTW_BACKWARD, FFTW_MEASURE /*FFTW_ESTIMATE*/);
	fftwf_execute(fft_p);
	fftwf_destroy_plan(fft_p);
	
	
}

//FFT on one vector of Y in row
void fftOneRow(complexF* Y, int cols, int row){
	
	fftwf_complex* org = (fftwf_complex*)&Y[row*cols];
	fftwf_complex* after = (fftwf_complex*)&Y[row*cols];

	fftwf_plan fft_p = fftwf_plan_dft_1d(cols, org, after, FFTW_FORWARD, FFTW_MEASURE /*FFTW_ESTIMATE*/);
	fftwf_execute(fft_p);
	fftwf_destroy_plan(fft_p);
//	fftw_free(org);fftw_free(after);
}

void numSyms(std::string in_file1, int cols) {
	std::ifstream infile;
	infile.open(in_file1.c_str(), std::ifstream::binary);
	infile.seekg(0, infile.end);
	size_t num_tx_samps = infile.tellg()/sizeof(complexF);
	infile.seekg(0, infile.beg);
	num_syms = std::ceil((float)num_tx_samps/(float)(cols-1));
	infile.close();
}

//Element by element multiplication
void matrixMultThenSum(complexF* Y, complexF* Hconj, complexF* Yf, int rows, int cols ){
	//Y x conj(H) -> then sum all rows into elements in Yf
	//Y = 16x1023
	//conjH = 16x1023
	for (int i = 0; i<rows; i++){
		for(int j=0; j<cols-1; j++){
			float Yreal = Y[i*(cols-1)+j].real;
			float Yimag = Y[i*(cols-1)+j].imag;
			float Hreal = Hconj[i*(cols-1)+j].real;
			float Himag = Hconj[i*(cols-1)+j].imag;
			//(a+bi)(c+di) = a*c - b*d + (bc + ad)i
			if(i==0){
				Yf[j].real = 0;
				Yf[j].imag=0;
			}
			
			Yf[j].real=Yf[j].real+(Yreal*Hreal - Yimag*Himag);
			Yf[j].imag=Yf[j].imag+(Yreal*Himag + Yimag*Hreal);	
		}
	}
	
} 

// Gives Sum of Hsqrd = |H|^2 as a 1x1023 vector
void findDistSqrd(complexF* H, complexF* Hsqrd, int rows, int cols){
	//initialize first row since Hsqrd currently holds X
	for (int j = 0; j<cols; j++){
		//|H|^2 = real^2 + imag^2
		//Sum of |H|^2 is summing all elements in col j
		Hsqrd[j].real = (H[j].real*H[j].real)+ (H[j].imag*H[j].imag);
		Hsqrd[j].imag =0;
	}
	
	for (int i = 1; i<rows; i++){  
		for (int j = 0; j<cols; j++){
			//|H|^2 = real^2 + imag^2
			//Sum of |H|^2 is summing all elements in col j
			Hsqrd[j].real = Hsqrd[j].real+ (H[i*cols + j].real*H[i*cols + j].real)+ (H[i*cols + j].imag*H[i*cols + j].imag);
		}
	}
	
}



//Divide matrix A by vector B -> element division
void divideOneRow(complexF * A, complexF * B, int cols, int row){
	int i=row;
	for(int j=0; j<cols; j++){
		float fxa = A[i*cols+j].real;
		float fxb = A[i*cols+j].imag;
		float fya = B[j].real;
		float fyb = B[j].imag;
		A[i*cols+j].real=((fxa*fya + fxb*fyb)/(fya*fya+fyb*fyb));
		A[i*cols+j].imag=((fxb*fya - fxa*fyb)/(fya*fya + fyb*fyb));	
	}
	
}

//Finds |H|^2 and H*=Hconj, rows=16 cols=1024
void firstVector(complexF* Y, complexF* Hconj, complexF* X, int rows, int cols, int iter = 0){
	//Read in X vector -> 1x1023
	matrix_readX(X, cols-1);
	//printOutArr(X, 1, cols-1);
	for (int i = 0; i<rows; i++){  
		for (int j = 0; j<cols; j++){
			Y[i*cols + j].real=0;
			Y[i*cols + j].imag=0;
			if(j<cols-1){
				Hconj[i*(cols-1)+j].real=0;
				Hconj[i*(cols-1)+j].imag=0;
			}
		}
	}
	
	//Do pre FFT bc library doesn't work first time
	//fftOneRow(Y, cols, 0);
	
	//Read in Y (get rid of prefix)
	/*
	if (iter < numberOfSymbolsToTest) {
		buffPtr->readNextSymbol(Y, iter);
	} else {
		buffPtr->readLastSymbol(Y);
	}
	*/
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	
	for(int row=0; row<rows; row++){
		//FFT one row 
		fftOneRow(Y, cols, row);
	}
	if(timerEn){
		finish = clock();
		fft[iter] = fft[iter] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	for(int row=0; row<rows; row++){
		//Drop first element and copy it into Hconj
		memcpy(&Hconj[row*(cols-1)], &Y[row*cols+1], (cols-1)* sizeof (*Y));
		
		//shift the row
		//shiftOneRow(Hconj, cols-1, row);
		
		//Divide FFT(Y) by X
		divideOneRow(Hconj, X, cols-1, row);
	}
	
	//take conjugate of H
	
	for (int i = 0; i<rows; i++){  
		for (int j = 0; j<cols-1; j++){
			Hconj[i*(cols-1) + j].imag = -1*Hconj[i*(cols-1) + j].imag;
		}
	}
	
	//Now Hconj holds H
	//Save |H|^2 into X
	findDistSqrd(Hconj,X,rows, cols-1);
	
	if(timerEn){
		finish = clock();
		decode[iter] = decode[iter] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
}

void doOneSymbol(complexF* Y, complexF* Hconj, complexF* Hsqrd,int rows, int cols, int it){
	
	if(it==numberOfSymbolsToTest-1){
		//read in 16x1024
		buffPtr->readLastSymbol(Y);
	}
	else{
		//read in 16x1024
		buffPtr->readNextSymbol(Y, it);
	}
	//printOutArr(Y, 1, cols);
	complexF* Yf = 0;
	Yf = (complexF*)malloc((cols-1)* sizeof (*Yf));
	complexF* Ytemp = 0;
	Ytemp = (complexF*)malloc(rows*(cols-1)*sizeof(*Ytemp));
	
	
	
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	
	for(int row=0; row<rows; row++){
		//FFT one row 
		fftOneRow(Y, cols, row);
	}
	if(timerEn){
		finish = clock();
		fft[it] = fft[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	for(int row=0; row<rows; row++){
		memcpy(&Ytemp[row*(cols-1)], &Y[row*cols+1], (cols-1)* sizeof (*Y));
		//shiftOneRow(Ytemp, cols-1, row);
	}
	
	//Find sum of YH* -> 1x1023
	matrixMultThenSum(Ytemp,Hconj,Yf, rows, cols);

	//Divide YH* / |H|^2
	//divideOneRow(Yf, Hsqrd, cols-1, 0);
	for (int j = 0; j < cols - 1; j++) {
		Yf[j].real = Yf[j].real/Hsqrd[j].real;
		Yf[j].imag = Yf[j].imag/Hsqrd[j].real;
	}
	shiftOneRow(Yf, cols-1, 0);
	if(timerEn){
		finish = clock();
		decode[it] = decode[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if (it <= 1) {
		outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
	} else {
		outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::app);
	}
	outfile.write((const char*)Yf, (cols-1)*sizeof(*Yf));
	outfile.close();

	/*
	if(testEn){
		printOutArr(Yf, 1, cols-1);
	}
	*/
	
	free(Yf);
}

void addPrefix(complexF* Y, complexF* dY, int rows, int cols) {
	
	for (int row = 0; row < rows; row++) {
		memcpy(&Y[row*(cols + prefix)], &dY[row*cols + (cols - prefix)], prefix*sizeof(*dY));
		memcpy(&Y[row*(cols + prefix) + prefix], &dY[row*cols], cols*sizeof(*dY));
	}
	
}

void rotCube(complexF* X, int rows, int cols, int users) {
	complexF* temp;
	temp = (complexF *)calloc(rows*users*cols,sizeof(*temp));
	
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			for (int user = 0; user < users; user++) {
				temp[col*users*rows + row*users + user] = X[user*rows*cols + row*cols + col];
			}
		}
	}
	memcpy(X, temp, rows*cols*users*sizeof(*X));
	free(temp);
}

void createZeroForcingMatrix(complexF* H, complexF* X, int rows, int cols, int users) {
	float alpha = 1, beta = 0;
	int info, lwork = users*users;
	char uplo = 'U';
	complexF* tempH;
	complexF* work;
	int* ipiv;
	tempH = (complexF *)malloc(users*users*sizeof(*tempH));
	ipiv = (int *)malloc(users*sizeof(*ipiv));
	work = (complexF *)malloc(lwork*sizeof(*work));
	
//	std::cout << "X before rot: " << X[0].real << ", " << X[1].real << ", " << X[2].real << ", " << X[3].real << std::endl;
	rotCube(X, rows, std::max(1,cols-1), users);
	
//	std::cout << "X after rot: " << X[0].real << ", " << X[1].real << ", " << X[2].real << ", " << X[3].real << ", " << X[4].real << ", " << X[5].real << std::endl;
	
	for (int i = 0; i < 6; i++) {
		std::cout << "( " << X[i].real << ", " << X[i].imag << ")";
	}
	std::cout << "\n";
	
	for (int col = 0; col < std::max(1,cols-1); col++) {
		cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, users, users, rows, &alpha, (float *)&X[col*rows*users], users, (float *)&X[col*rows*users], users, &beta, (float *)tempH, users);
		cgetrf_(&users, &users, tempH, &users, ipiv, &info);
		cgetri_(&users, tempH, &users, ipiv, work, &lwork, &info);
		cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, rows, users, users, &alpha, (float *)&X[col*rows*users], users, (float *)tempH, users, &beta, (float *)&H[col*rows*users], rows);
	}
	
	
	free(tempH);
	free(ipiv);
	free(work);
}

void multiplyWithChannelInv(complexF* HX, complexF* X, complexF* H, int rows, int cols, int users) {
	float alpha = 1, beta = 0;
	complexF* vec = 0;
	vec = (complexF *)malloc(users*sizeof(*vec));
	
	for (int i = 0; i < cols-1; i++) {
		/*
		for(int user = 0; user < users; user++) {
			vec[user] = X[user*(cols-1) + i];
		}
		*/
		cblas_cgemv(CblasColMajor, CblasNoTrans, rows, users, &alpha, (float *)&H[rows*users*i], rows, (float *)&X[i], cols-1, &beta, (float *)&HX[i], cols-1);
	}
	free(vec);
}


void modRefSymbol(complexF* Y, complexF* X, int cols) {
	
	matrix_readX(X, cols-1);
	complexF* dY;
	dY = (complexF *)calloc(cols,sizeof(*dY));
	float* work;
	work = (float *)malloc(cols*sizeof(*work));
	char norm = 'M';
	int temp = 1;
	float maxval;
	
	memcpy(&dY[1], X, (cols-1)*sizeof(*X));
		
	ifftShiftOneRow(dY, cols, 0);
	ifftOneRow(dY, cols, 0);
	
	
	maxval = 1/clange_(&norm, &cols, &temp, dY, &cols, work);
	cblas_csscal(cols, maxval, (float *)dY, 1);
		
	addPrefix(Y, dY, 1, cols);
	free(dY);
	free(work);
}


void modOneSymbol(complexF* Y, complexF* H, complexF* X, int rows, int cols, int users, bool chanMultiply = false) {
	if (chanMultiply == true) {
		multiplyWithChannelInv(Y, H, X, rows, cols, users);
	} else {
		rows = users;
		memcpy(Y, X, rows*(cols-1)*sizeof(*X));
		
	}
	complexF* dY;
	dY = (complexF *)calloc(rows*cols,sizeof(*dY));
	float* work;
	work = (float *)malloc(cols*sizeof(*work));
	char norm = 'M';
	int temp = 1;
	float maxval;
	
	for (int row = 0; row < rows; row++) {
	//	dY[row*cols].real = 0;
	//	dY[row*cols].real = 0;
		memcpy(&dY[row*cols + 1], &Y[row*(cols-1)], (cols-1)*sizeof(*Y));
		
		ifftShiftOneRow(dY, cols, row);
		ifftOneRow(dY, cols, row);
		/*
		int temp = cblas_icamax(cols, (float *)&dY[row*cols], 1);
		std::cout << temp << std::endl;
		float temp2 = 1/std::sqrt(dY[temp + row*cols].real*dY[temp + row*cols].real + dY[temp + row*cols].imag*dY[temp + row*cols].imag);
		cblas_csscal(cols, temp2, (float *)&dY[row*cols], 1);
		*/
		maxval = 1/clange_(&norm, &cols, &temp, &dY[row*cols], &cols, work);
		//std::cout << maxval << std::endl;
		cblas_csscal(cols, maxval, (float *)&dY[row*cols], 1);
	}
	
	addPrefix(Y, dY, rows, cols);
	free(dY);
	free(work);
}

#endif