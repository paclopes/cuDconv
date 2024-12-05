/*
 * Implementation of a Convolution Neural Network convolution layer.
 * Annex to the paper: 
 * Open Efficient CUDA Convolution Neural Network Inference Implementation
 * Paulo A. C. Lopes
 * Instituto de Engenharia de Sistemas e Computadores: Investigação e Desenvolvimento em Lisboa (INESC-ID)
 * Instituto Superior Técnico, Universidade de Lisboa
 * to be published
 */

#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
//#undef NDEBUG // Uncomment to make sure asserts are always checked.
#include <assert.h>
//#define PROFILE // Uncomment to enable profilling

// Edit the layer sizes bellow
const int batch_size = 10;
const int in_channels = 256;
const int out_channels = 256;
const int height = 16;
const int width = 16;

const int kernel_h = 3;
const int kernel_w = 3;

// div ceiling
// divC(a, b) = ceil((float)a / b))
#define divC(a, b) (((a) + (b) - 1)/(b))
/*
__device__ int divC(int a, int b) {
	return (a + b - 1)/b;
}*/

typedef float Dtype;
typedef float4 Dtypex;
const int LS = sizeof(Dtypex)/sizeof(Dtype);
#define DtypexPlus

const int warpSize = 32;

// Tile 1 to tile 5 sizes

const int M1 = 32;		// Tile
const int P1 = 8*9;     // Tile
const int P0 = 128*9;
const int N1 = 2*128;	// Tile 
const int M5 = 1;		// loop
const int N5 = 2*LS;    // loop  
const int M3 = 4;	    // lane
const int N4 = M5*N5;
const int N3 = 8*N4;    // lane
const int M2 = 8*M3;    // loop
const int N2 = 1*N3;    // loop
const int P2 = 9;
const int n_threads = M1*N1/(M2/M3*N2/N3*N4);

const int KS = kernel_h*kernel_w;
const int IS = height*width;
const int IC = in_channels;
const int OC = out_channels;
const int BS = batch_size;
const int M = out_channels;
const int P = kernel_h*kernel_w*in_channels;
const int N = batch_size*height*width;
const bool ex1 = N1 < IS;
const int N1z = (ex1)?width:0;
const int N1x = N1 + 2*N1z;

const int Gx = divC(M, M1);
const int Gy = divC(N, N1);
const int Gz = divC(P, P0);
typedef Dtype out_regs_t[M2/M3][N2/N3][N4];

__device__ int mutex[Gx][Gy];

__device__ void inline nanosleep(unsigned x) {
#if __CUDA_ARCH__ >= 700
	__nanosleep(x);
#else
	unsigned start_clock = clock();
	while (clock() - start_clock > x);
#endif
}

__device__ void get_global(int i, int j) {
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		while (atomicExch(&mutex[i][j], 1)) nanosleep(100);
	__syncthreads();
}

__device__ void free_global(int i, int j) {
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		atomicExch(&mutex[i][j], 0);
}

__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

// Helper functions to work with vector types

__device__ void set(float &x, int i, float v)  {
	x = v;
}

__device__ void set(float2 &x, int i, float v)  {
	switch (i) {
		case 0: x.x = v; break;
		case 1: x.y = v; break;
	}
}

__device__ void set(float4 &x, int i, float v) {
	switch (i) {
		case 0: x.x = v; break;
		case 1: x.y = v; break;
		case 2: x.z = v; break;
		case 3: x.w = v; break;
	}
}

__device__ Dtypex zero() {
	Dtypex y;
	for(int i = 0; i < LS; i++)
		set(y, i, 0);
	return y;
}

__device__ float get(float x, int i) {
	return x;
}

__device__ float get(float2 x, int i) {
	float v;
	switch (i) {
		case 0: v = x.x; break;
		case 1: v = x.y;
	}
	return v;
}

__device__ float get(float4 x, int i) {
	float v;
	switch (i) {
		case 0: v = x.x; break;
		case 1: v = x.y; break;
		case 2: v = x.z; break;
		case 3: v = x.w;
	}
	return v;
}

// Addition for vector types (Dtypex)

#ifdef DtypexPlus
__device__ Dtypex inline operator+(Dtypex a, Dtypex b) {
    Dtypex v;
    for(int i = 0; i < LS; i++)
        set(v, i, get(a, i) + get(b, i));
    return v;
}
#endif

// Helper functions to load and store from memory

template <typename Dtype>
__device__ inline Dtype load(const Dtype *A, const int i)
{	
	return *(A + i);
}

template <typename Dtype>
__device__ inline Dtypex loadx(const Dtype *A, const int i)
{	
	return *(Dtypex *)(A + i);
}

template <typename Dtype>
__device__ inline void  storex(Dtype *A, const Dtypex value, const int i)
{
	*(Dtypex *)(A + i) = value;
}

template <typename Dtype>
__device__ inline void addx(Dtype *A, const Dtypex value, const int i)
{
    Dtypex *Ax = (Dtypex *)(A+i);
    *Ax = *Ax + value;
}

// Helper functions for integer division and modulo

// Division Floor
// divF(a, b) = floor((float)a / b);
__device__ inline int divF(int a, int b)
{
	if (a >= 0)
		return a / b;
	else 
		return - divC(- a, b);
}

// Modulo
// mod(a, b) = a - divF(a, b) * b
__device__ inline int mod(int a, int b)
{
	int c = a % b;
	if (c >= 0)
		return c;
	else
		return c + b;
}

// Integer division and modulo implemented by multiplying by the inverse
__device__ inline void divx(const int a, const int b, int &q, int &m)
{
	const int bits = 16;
	const int b1 = (1 << bits) / b;	

	q = (b1 * a) >> bits; // quotient
	m = a - q * b;        // modulo
	if (m >= b) {         // correction
		m -= b;
		q++;
	}
}

// The struct intx represents an int formed by the sum of a compile time constant x and variable y.
// Used to distribute divisions by both values, when possible.
struct intx {
	int x, y;
	__device__ inline operator int() {
    	return x + y;
	}
	__device__ inline intx operator+(int z) {
		intx a;
		a.x = x;
		a.y = y + z;
    	return a;
	}

};

__device__ inline int add(int a, int b) {
	return a + b;
}

__device__ inline intx addx(int a, int b) {
	intx y = {a, b}; 
	return y;
}

// Distributes division by b through intx values

__device__ inline void divx(const intx a, const int b, intx &q, intx &m) {
	divx(a.x, b, q.x, m.x);
	divx(a.y, b, q.y, m.y);
	if (m >= b) { // m = m.x + m.y
		m.y -= b;
		q.y++;
	}
}

/********************************************************************************
 * matrix_tile
 *
 * Copies a tile of matrix from global to shared memory.
 *
 * tile:    matrix tile in shared memory
 * matrix:  matrix in global memory
 * i0, j0:  tile upper left matrix coordinates
 * M, N:    matrix size (rows, columns)
 * M1, N1:  tile size (rows, columns)
 * tId:     thread ID
 * n_threads: number of threads
 */

template <typename Dtype>
__device__ inline void matrix_tile(
        Dtype* tile,
        const Dtype* matrix,
        const int i0, 
        const int j0,
        const int M,
        const int N,
        const int M1,
        const int N1,
        const int tId,
        const int n_threads
        )
{
	#pragma unroll
	for(int m0 = 0; m0 < M1*N1; m0 += LS*n_threads) {
		intx i1i, j1i;
		auto m = addx(m0, LS*tId);
		divx(m, N1, i1i, j1i);
		int i = i0 + i1i;
		int j = j0 + j1i;

		Dtypex data = zero();
		if (i < M && j < N)
        	data = loadx(matrix, i*N + j);
		if (i1i < M1 && j1i < N1)
			storex(tile, data, i1i*N1 + j1i);
	}
}

/********************************************************************************
 * tensor_tile
 *
 * Copies a tile of tensor from global to shared memory.
 * The tensor with indexes ix, jx, kx, lx
 * is reshaped to a matrix with indexes 
 *	i = ix and j = jx*P*Q + kx*P + lx, if permute is false and
 *	i = jx and j = ix*P*Q + kx*P + lx, if permute is true
 *
 * tile:    tensor tile in shared memory
 * tensor:  tensor in global memory
 * i0, j0:  tile upper left matrix coordinates
 * M, N, P, Q:    tensor size
 * M1, N1:  tile size (rows, columns)
 * tId:     thread ID
 * n_threads: number of threads
 * permute: permute tensor coordinates?
 */

template <typename Dtype>
__device__ inline void tensor_tile(
        Dtype* tile,
        const Dtype* tensor,
        const int i0, // tile at iy = i0
        const int j0, // tile at jy*P + ky = j0
        const int M,
        const int N,
        const int P,
        const int Q,
        const int M1,
        const int N1,
        const int tId,
        const int n_threads,
		bool permute
        )
{
	#pragma unroll
	for(int m0 = 0; m0 < M1*N1; m0 += LS*n_threads) {
		auto m = addx(m0, LS*tId);
		intx i1i, j1i;
		divx(m, N1, i1i, j1i);
		intx i = i1i + i0;
		intx j = j1i + j0;
		// i; j > iy; jy, ky, ly
		int iy = i;
		intx jy, ky;
		divx(j, P*Q, jy, ky);
		int ix = iy, jx = jy, kx = ky;
		if (permute) { 
			ix = jy; jx = iy;
		}

		Dtypex data = zero();
		if (ix < M && jx < N && kx < P*Q)
        	data = loadx(tensor, ix*N*P*Q + jx*P*Q + kx);
		if (i1i < M1 && j1i < N1)
			storex(tile, data, i1i*N1 + j1i);
	}
}

/********************************************************************************
 * conv_tile
 *
 * Computes a tile Y1 of the convolution of X with W, Y1 = W1*X1'.
 * 
 * v: output
 * W: W tile in shared memory
 * X: X tile in shared memory
 * i1, j1: tile 1 indexes
 * i2, j2: tile 2 indexes (from warp ID)
 * i4, j4: tile 4 indexes (from lane ID)
 * tId: thread ID
 * n1z: offset to allow larger input image (more lines) than the output image as required by convolution
 */

__device__ void inline conv_tile(out_regs_t &v, Dtype *W, Dtype *X, int i1, int j1, int i2, int j2, int i4, int j4, int tId, int n1z){
	#pragma unroll 
	for (int k2 = 0; k2 < divC(P1, P2); k2++) {
		#pragma unroll 
		for(int j3 = 0; j3 < N2/N3; j3++) {
			Dtype bx[M5+2][N5+2];
			//const int j = n & l1 & c1 & l0 & c0
			auto j = addx(j3*N3, j1*N1 + j2*N2 + j4*N4);
			intx n, px, m1, m0, l1, c1;
			divx(j, IS, n, px);
			divx(px, N4, m1, m0);
			divx(m1, width/N5, l1, c1);
			const int c0 = 0;
			const int c = c1*N5 + c0;
			// padding predicates
			const bool left = (c == 0);
			const bool right = (c + N5 == width);
			const bool top = (l1 == 0);
			const bool bottom = (l1 == height/M5 - 1);

			// load X to b
			#pragma unroll
			for(int l0 = 0; l0 < M5+2; l0++) {
				if (l0 == 0 && top || l0 == M5+1 && bottom) {
					for(int c0 = 0; c0 < N5+2; c0++) 
						 bx[l0][c0] = 0;
				} else {
					const int l = l1*M5 + l0 - 1;
					const int jx = n*IS + l*width + c - N1*j1;
					if (left) bx[l0][0] = 0;
					else bx[l0][0] = load(X, k2*N1x + jx - 1 + n1z);
					for(int c0 = 0; c0 < N5; c0 += LS) {
						Dtypex b = loadx(X, k2*N1x + jx + c0 + n1z);
						for(int c2 = 0; c2 < LS; c2++)
							bx[l0][c0 + c2 + 1] = get(b, c2);
					}
					if (right) bx[l0][N5+1] = 0;
					else bx[l0][N5+1] = load(X, k2*N1x + jx + N5 + n1z);
				}
			}

			// load W to a and calculate convolution
			#pragma unroll
			for(int i3 = 0; i3 < M2/M3; i3++) {
				Dtypex ax;
				#pragma unroll
				for(int k3 = 0; k3 < P2; k3++) {
					const int k = k2*P2 + k3;
					const int i = i2*M2 + i3*M3 + i4;
					const int r = mod(k, LS);
					if (r == 0 || k3 == 0) { 
						ax = loadx(W, i*P1 + k - r);
					};
					Dtype a = get(ax, r);

					#pragma unroll
					for(int j5 = 0; j5 < N4; j5++) {
						const int kl = k3 / kernel_w;
						const int kc = k3 % kernel_w;
						const int l = j5 / N5;
						const int c = j5 % N5;
						v[i3][j3][j5] += a * bx[l - kl + 2][c - kc + 2];
					}
				}
			}
		}
	}
}

/********************************************************************************
 * store_tile
 *
 * Stores or adds the output tile to global memory.
 * 
 * output: output tensor Y in global memory
 * v: conv_tile output in registers
 * i1, j1: tile 1 indexes
 * i1, j2: tile 2 indexes
 * i4, j4: tile 4 indexes 
 */

__device__ inline void store_tile(Dtype *output, out_regs_t &v, int i1, int j1, int i2, int j2, int i4, int j4) {
	if (P0 < P) get_global(i1, j1);
    #pragma unroll
	for(int i3 = 0; i3 < M2/M3; i3++)
		#pragma unroll
        for(int j3 = 0; j3 < N2/N3; j3++) 
			#pragma unroll
			for(int j5 = 0; j5 < N4; j5 += LS) {
				// determine matrix coordinates
				int i = i1*M1 + i2*M2 + i3*M3 + i4;
				auto j = addx(j3*N3 + j5, j1*N1 + j2*N2 + j4*N4);
				// j = n, l1, c1, l0, c0 = n*IS + l1*width/N5*N4 + c1*N4 + l0*N5 + c0
				
				Dtypex vx;
				for(int j6 = 0; j6 < LS; j6++)
					set(vx, j6, v[i3][j3][j5 + j6]);

				// determine tensor coordinates
				intx n, px1, m0, m1, l1, l0, c1, c0;
				divx(j, IS, n, px1);
				divx(px1, N4, m1, m0);
				divx(m1, width/N5, l1, c1);
				divx(m0, N5, l0, c0);
				const int l = l1*M5 + l0;
				const int c = c1*N5 + c0;
				const int px = l*width + c;
				const int co = i;
				
				// store in output tensor
				if (n < BS && co < OC) {
					if (P0 < P) addx(output, vx, n*OC*IS + co*IS + px);
					else storex(output, vx, n*OC*IS + co*IS + px);
				}
			}
	if (P0 < P) free_global(i1, j1);
}

#ifdef PROFILE
/*
 * Allow profilling the code by storing the clock register value when each thread block enters a given code section.
 * The clocks values are store to "profile.txt" together with SM and thread block ID.
 */
#define N_CLOCKS (2 + 2*P0/P1)
const unsigned int total_blocks = Gx*Gy*Gz;
unsigned clocks_h[total_blocks*N_CLOCKS];
unsigned SM_h[total_blocks];
__device__ unsigned pclocks[total_blocks*N_CLOCKS];
__device__ unsigned SM[total_blocks];
#define PROFILE_KERNEL_SETUP \
    unsigned blockId = (blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;\
	int kx = 0;
#define GET_TIME(x) \
    __syncthreads();\
    if (tId == 0) {\
        pclocks[blockId*N_CLOCKS + x] = clock();\
    }\
    __syncthreads();
#define STORE_TIME \
    if (tId == 0) {\
        SM[blockId] = get_smid();\
    }
#define PROFILE_PRINT \
    checkCudaErrors(cudaMemcpyFromSymbol(clocks_h, pclocks, total_blocks*N_CLOCKS*sizeof(unsigned)));\
    checkCudaErrors(cudaMemcpyFromSymbol(SM_h, SM, total_blocks*sizeof(unsigned)));\
    std::ofstream file("profile.txt");\
    for(int w = 0; w < total_blocks; w++){\
        file << SM_h[w]  << ", " << w << ", ";\
        for(int i = 0; i < N_CLOCKS; i++){\
            file << clocks_h[w*N_CLOCKS + i] << ", ";\
        }\
        file << std::endl;\
    }\
    file.close();
#else
#define PROFILE_KERNEL_SETUP
#define GET_TIME(X)
#define STORE_TIME
#define PROFILE_PRINT
#endif

/********************************************************************************
 * gpu_convolution_kernel
 *
 * GPU kernel that implements CNN inference convolution layer.
 * 
 */

template <typename Dtype>
__global__ void gpu_convolution_kernel(Dtype* output, 
		const Dtype* weights,
		const Dtype* input)
{
	PROFILE_KERNEL_SETUP;
	const int i1 = blockIdx.x;
	const int j1 = blockIdx.y;
	const int k0 = blockIdx.z;
	const int wId = threadIdx.y;
	const int lId = threadIdx.x;
	const int tId = wId * warpSize + lId;
    const int n1z = (j1 == 0) ? 0 : N1z;
	const int P1x = P1/KS;

	__shared__ Dtype W[M1*P1];
	__shared__ Dtype X[P1x*N1x];

	const int i2 = wId / (N1/N2);
	const int j2 = wId % (N1/N2);
	const int i4 = lId / (N3/N4);
	const int j4 = lId % (N3/N4);

	GET_TIME(kx++);

	out_regs_t v;
	#pragma unroll
    for(int i3 = 0; i3 < M2/M3; i3++)
		#pragma unroll
        for(int j3 = 0; j3 < N2/N3; j3++)
			#pragma unroll
        	for(int j5 = 0; j5 < N4; j5++)
           		v[i3][j3][j5] = 0;

	#pragma unroll 1 
	for(int k1 = k0*P0/P1; k1 < min((k0+1)*P0/P1, P/P1); k1++) {
    	matrix_tile(W, weights, i1*M1, k1*P1, M, P, M1, P1, tId, n_threads);
    	tensor_tile(X, input, k1*P1x, j1*N1 - n1z, BS, IC, width, height, P1x, N1x, tId, n_threads, true);
		
		GET_TIME(kx++);
		__syncthreads();

		conv_tile(v, W, X, i1, j1, i2, j2, i4, j4, tId, n1z);
		GET_TIME(kx++);

		__syncthreads();
	}

	store_tile(output, v, i1, j1, i2, j2, i4, j4);
	GET_TIME(kx++);

	STORE_TIME;
}

// Helper function to fill a tensor with random numbers from -1 to 1.

template <typename Dtype>
void fill(Dtype *x, int n)
{
	for(int i = 0; i < n; i++)
		x[i] = 2*float(rand())/RAND_MAX-1;	
}

/********************************************************************************
 * check_convolution
 * 
 * Checks the conv layer output with a an equivalent CPU implementation and reports possible errors.
 *
 */

template <typename Dtype>
int check_convolution(const Dtype* output,
        const Dtype* weights,
        const Dtype* input,
        const int batch_size,
        const int in_channels, const int out_channels,
        const int height, const int width,
        const int kernel_hx, const int kernel_wx)
{
	int errors = 0;

	for(int n = 0; n < batch_size; n++)
		for(int oc = 0; oc < out_channels; oc++)
			for(int i = 0; i < height; i++)
				for(int j = 0; j < width; j++) {
					Dtype y = 0;
					for(int ic = 0; ic < in_channels; ic++)
						for(int ki = 0; ki < kernel_hx; ki++)
							for(int kj = 0; kj < kernel_wx; kj++){
								int ix = i + 1 - ki;
								int jx = j + 1 - kj;
								if (ix >= 0 && ix < height && jx >=0 && jx < width)
									y += weights[ (((oc * in_channels + ic) * kernel_hx + ki) * kernel_wx + kj)] * 
										input[ ((n * in_channels + ic) * height + ix) * width + jx];
							}
					Dtype y0 = output[((n * out_channels + oc) * height + i) * width + j];
					if (abs(y0 - y) > 0.01 || isnan(y0)) {
						errors++;
						if (errors < 10) 
							std::cout << "out value: " << y0 << " correct value: " << y << std::endl;
					}
				};
	return errors;
}

Dtype
	input_h[batch_size*in_channels*height*width],
	weights_h[out_channels*in_channels*kernel_h*kernel_w],
	output_h[batch_size*out_channels*height*width];	

/********************************************************************************
 * main
 *
 * Fill input and weights with random numbers, then times the kernel and checks the output.
 *
 */

int main() {
	assert(P2 == KS);
	assert(P % P1 == 0);
	assert(P1 % P2 == 0);
	assert(M3*N3 / N4 == warpSize);
	assert(M1 >= M2);
	assert(N1 >= N2);
	assert(N5 % LS == 0);
	assert(M1*N1 % LS == 0);

	int kernel_runs = 100+1;

	std::cout << "Filling matrices." << std::endl;

	fill(input_h, batch_size*in_channels*height*width);
	fill(weights_h, out_channels*in_channels*kernel_h*kernel_w);
	fill(output_h, batch_size*out_channels*height*width);

	Dtype *output, *input, *weights;	

	checkCudaErrors(cudaMalloc(&output, batch_size*out_channels*height*width*sizeof(Dtype)));
	checkCudaErrors(cudaMalloc(&input, batch_size*in_channels*height*width*sizeof(Dtype)));
	checkCudaErrors(cudaMalloc(&weights, out_channels*in_channels*kernel_h*kernel_w*sizeof(Dtype)));

	checkCudaErrors(cudaMemcpy(output, output_h, batch_size*out_channels*height*width*sizeof(Dtype), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input, input_h, batch_size*in_channels*height*width*sizeof(Dtype), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(weights, weights_h, out_channels*in_channels*kernel_h*kernel_w*sizeof(Dtype), cudaMemcpyHostToDevice));

    dim3 blocks(Gx, Gy, Gz);
    dim3 threads(warpSize, n_threads/warpSize);
    std::cout  << "blocks: " << blocks.x << ", " << blocks.y << ", " << blocks.z <<" " << "threads: " << threads.x << ", " << threads.y << std::endl;

	std::vector <float> times(kernel_runs);

	cudaFuncSetAttribute(gpu_convolution_kernel<Dtype>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared); 
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); 
	std::cout << "Shared memory size: " << prop.sharedMemPerMultiprocessor/1024 << " kiB" << std::endl;
	std::cout << "L2 cache size: " << prop.l2CacheSize/1024 << " kiB" << std::endl;

	int *mutex_d;
	checkCudaErrors(cudaGetSymbolAddress((void **)&mutex_d, mutex)); 

	for(int n = 0; n < kernel_runs; n++) {
		auto start = std::chrono::high_resolution_clock::now();
	
   		checkCudaErrors(cudaMemset(output, 0, batch_size*out_channels*height*width*sizeof(Dtype)));
   		checkCudaErrors(cudaMemset(mutex_d, 0, Gx*Gy*sizeof(int)));
	
   	 	gpu_convolution_kernel<Dtype><<<blocks, threads>>>(
				output,
				weights,
				input);

		checkCudaErrors(cudaDeviceSynchronize());

	    auto finish = std::chrono::high_resolution_clock::now();
	    float kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
		times[n] = kernel_time;
	}

	std::sort(times.begin(), times.end());
    std::cout << "The Kernel took " << times[(kernel_runs-1)/2] << " microseconds (median)." << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    float clock = deviceProp.clockRate / 1000; // Mhz
    std::cout << "The clock is " << clock << " MHz." << std::endl;

	PROFILE_PRINT;

	checkCudaErrors(cudaMemcpy(output_h, output, batch_size*out_channels*height*width*sizeof(Dtype), cudaMemcpyDeviceToHost));

	std::cout << "Errors: " << check_convolution(
                output_h,
                weights_h,
                input_h,
                batch_size,
                in_channels, out_channels,
                height, width,
                kernel_h, kernel_w
            ) << std::endl;

	checkCudaErrors(cudaFree(output));
	checkCudaErrors(cudaFree(input));
	checkCudaErrors(cudaFree(weights));
}
