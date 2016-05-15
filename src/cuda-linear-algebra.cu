/*
 ============================================================================
 Name        : cuda-linear-algebra.cu
 Author      : Alex Minnaar
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>


typedef struct{
	int width;
	int height;
	float* elements;
} Matrix;

#define BLOCK_SIZE 3

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


void MatMul(const Matrix A, const Matrix B, Matrix C)
{

	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width*A.height*sizeof(float);

	//allocate memory for matrix A on device
	cudaMalloc(&d_A.elements,size);
	//copy matrix A elements to memory allocated on device
	cudaMemcpy(d_A.elements, A.elements,size,cudaMemcpyHostToDevice);

	//do the same thing for matrix B
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;

	//allocate memory for matrix B on device
	cudaMalloc(&d_B.elements,size);
	//copy matrix B elements to memory allocated on device
	cudaMemcpy(d_B.elements, B.elements,size,cudaMemcpyHostToDevice);


	//allocate memory for matrix C on device - obviously nothing to copy
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	cudaMalloc(&d_C.elements,size);

	//define block size and grid size for kernel
	dim3 dimBlock(1,3);
	//dim3 dimGrid(B.width/dimBlock.x,A.height/dimBlock.y);

	//invoke kernel
	MatMulKernel<<<3, dimBlock>>>(d_A, d_B, d_C);

	//read matrix multiplication result from device
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	//Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}



//The ACTUAL kernel that performs matrix multiplication in the GPU
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{

	//each kernel thread computes one element of C
	float Cvalue=0;

	//therefore, each thread is needs to read the row of A and column of B for its element of C.
	//reading from global memory here - would be better if the row and column were stored in shared memory
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}


int main(void)
{

	float *a_els = new float[9];
	for(int i = 0; i<9; ++i)
		a_els[i]=2.0;

	float *b_els = new float[9];
	for(int i = 0; i<9; ++i)
		b_els[i]=2.0;

	Matrix m_a;
	m_a.height=3;
	m_a.width=3;
	m_a.elements = a_els;

	Matrix m_b;
	m_b.height=3;
	m_b.width=3;
	m_b.elements = a_els;

	float *c_els=new float[9];

	Matrix m_c;
	m_c.height = 3;
	m_c.width = 3;
	m_c.elements=c_els;

	MatMul(m_a,m_b,m_c);


	for(int i =0; i<9;i++)
		printf("%f \n",m_c.elements[i]);



	delete[] a_els;
	delete[] b_els;
	delete[] c_els;




	return 0;
}



