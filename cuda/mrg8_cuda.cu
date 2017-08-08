/*
 * mrg8_cuda.cu
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: June 29, 2017
 *      Author: Yusuke
 */

#include "mrg8_cuda.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <ctime>

using namespace std;

// #define TNUM (2048 * 64 * 4)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8::mrg8(): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(1927846343), iseed(0), isJumpMatrix(false)
{
    JUMP_MATRIX = new uint32_t[8 * 8 * 247];
    JUMP_MATRIX_R = new uint32_t[8 * 8 * 247];
    JUMP_MATRIX_8s_32 = new uint32_t[8 * 8 * 4];
    JUMP_MATRIX_8s_64 = new uint32_t[8 * 8 * 8];

	mcg64ni();
	read_jump_matrix();
    reverse_jump_matrix();
    set_jump_matrix_8s_32();
    set_jump_matrix_8s_64();

    cudaMalloc((void **)&d_JUMP_MATRIX, sizeof(uint32_t) * 8 * 8 * 247);
    cudaMalloc((void **)&d_JUMP_MATRIX_8s_32, sizeof(uint32_t) * 8 * 8 * 4);
    cudaMalloc((void **)&d_JUMP_MATRIX_8s_64, sizeof(uint32_t) * 8 * 8 * 8);

    cudaMemcpy((int *)d_JUMP_MATRIX, (int *)JUMP_MATRIX_R, sizeof(uint32_t) * 8 * 8 * 247, cudaMemcpyHostToDevice);
    cudaMemcpy((int *)d_JUMP_MATRIX_8s_32, (int *)d_JUMP_MATRIX_8s_32, sizeof(uint32_t) * 8 * 8 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy((int *)d_JUMP_MATRIX_8s_64, (int *)JUMP_MATRIX_8s_64, sizeof(uint32_t) * 8 * 8 * 8, cudaMemcpyHostToDevice);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8::mrg8(const uint32_t seed_val): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(1927846343), iseed(seed_val), isJumpMatrix(false)
{
    JUMP_MATRIX = new uint32_t[8 * 8 * 247];
    JUMP_MATRIX_R = new uint32_t[8 * 8 * 247];
    JUMP_MATRIX_8s_32 = new uint32_t[8 * 8 * 4];
    JUMP_MATRIX_8s_64 = new uint32_t[8 * 8 * 8];

	mcg64ni();
	read_jump_matrix();
    reverse_jump_matrix();
    set_jump_matrix_8s_32();
    set_jump_matrix_8s_64();

    cudaMalloc((void **)&d_JUMP_MATRIX, sizeof(uint32_t) * 8 * 8 * 247);
    cudaMalloc((void **)&d_JUMP_MATRIX_8s_32, sizeof(uint32_t) * 8 * 8 * 4);
    cudaMalloc((void **)&d_JUMP_MATRIX_8s_64, sizeof(uint32_t) * 8 * 8 * 8);

    cudaMemcpy((int *)d_JUMP_MATRIX, (int *)JUMP_MATRIX_R, sizeof(uint32_t) * 8 * 8 * 247, cudaMemcpyHostToDevice);
    cudaMemcpy((int *)d_JUMP_MATRIX_8s_32, (int *)JUMP_MATRIX_8s_32, sizeof(uint32_t) * 8 * 8 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy((int *)d_JUMP_MATRIX_8s_64, (int *)JUMP_MATRIX_8s_64, sizeof(uint32_t) * 8 * 8 * 8, cudaMemcpyHostToDevice);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//calculate (x[0]*y[0] + x[1]*y[1]+...+x[7]*y[7]) mod (2^31-1)
// where x[i] < 2^31 and y[i] < 2^31, i = 0, ..., 7
// Let:
//     s1 = x[0] * y[0] + ... + x[3] * y[3]
//     s2 = x[4] * y[4] + ... + x[7] * y[7]
//     mask = 2^31 -1
// Note that s1 < 2^64 and s2 < 2^64
// (s1 + s2) mod (2^31 -1) = (s1 mod (2^31-1) + s2 mod (2^31-1)) mod (2^31-1)
// s1 = (s1 & mask) + 2^31 * (s1 >> 31) --> s1 mod (2^31-1) = (s1 & mask) + (s1 >> 31)
// s2 = (s2 & mask) + 2^31 * (s2 >> 31) --> s2 mod (2^31-1) = (s2 & mask) + (s2 >> 31)
uint32_t mrg8::bigDotProd(const uint32_t x[8], const uint32_t y[8]) const
{
	uint64_t s, s1, s2;
	s1 = 0;
    s2 = 0;
    s = 0;
	for (int q = 0; q < 4; ++q) {
		s1 += uint64_t(x[q]) * y[q];
		s2 += uint64_t(x[4 + q]) * y[4 + q];
	}
	s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + ( s2 >> 31);
    s = ((s & MASK) + (s >> 31));
	return ((s & MASK) + (s >> 31));
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reading pre-calculated jump matrix from file
// A^(2^j) mod (2^31 - 1) for j = 0, 1, ..., 246 where A is the 8x8 one-step jump matrix
// Each matrix is in column order, that is (A^(2^j))(r,c) = JUMP_MATRIX[j * 64 + r + 8*c]
void mrg8::read_jump_matrix()
{
    if (isJumpMatrix) return;
    
	std::ifstream infile;
	infile.open("jump_matrix.txt");
	if (infile.fail()) {
		std::cerr << "jump_matrix.txt could not be opened! Terminating!"<<std::endl;
		exit(EXIT_FAILURE);
	}

	uint32_t t;
	for (int k = 0; k < 8 * 8 * 247; ++k) {
		infile >> t;
		JUMP_MATRIX[k] = t;
	}
    infile.close();
    isJumpMatrix = true;
}

void mrg8::reverse_jump_matrix()
{
    int i, j, k;
    for (i = 0; i < 247; ++i) {
        for (j = 0; j < 8; ++j) {
            for (k = 0; k < 8; ++k) {
                JUMP_MATRIX_R[i * 64 + (7 - j) * 8 + (7 - k)] = JUMP_MATRIX[i * 64 + j + k * 8];
            }
        }
    }
}

void mrg8::set_jump_matrix_8s_32()
{
    int i, j;
    /* A^8 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_32[i + 32 * j] = JUMP_MATRIX_R[3 * 64 + i * 8 + j];
        }
    }
    /* A^16 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_32[8 + i + 32 * j] = JUMP_MATRIX_R[4 * 64 + i * 8 + j];
        }
    }
    /* A^24 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_32[16 + i + 32 * j] = bigDotProd(JUMP_MATRIX_R + 4 * 64 + i * 8, JUMP_MATRIX_8s_32 + j * 32); // A^16 * A^8
        }
    }
    /* A^32 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_32[24 + i + 32 * j] = JUMP_MATRIX_R[5 * 64 + i * 8 + j];
        }
    }
}

void mrg8::set_jump_matrix_8s_64()
{
    int i, j;
    int nrow = 64;
    /* A^8 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[i + nrow * j] = JUMP_MATRIX_R[3 * 64 + i * 8 + j];
        }
    }
    /* A^16 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[8 + i + nrow * j] = JUMP_MATRIX_R[4 * 64 + i * 8 + j];
        }
    }
    /* A^24 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[16 + i + nrow * j] = bigDotProd(JUMP_MATRIX_R + 4 * 64 + i * 8, JUMP_MATRIX_8s_64 + j * nrow); // A^16 * A^8
        }
    }
    /* A^32 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[24 + i + nrow * j] = JUMP_MATRIX_R[5 * 64 + i * 8 + j];
        }
    }
    /* A^40 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[32 + i + nrow * j] = bigDotProd(JUMP_MATRIX_R + 5 * 64 + i * 8, JUMP_MATRIX_8s_64 + j * nrow); // A^32 * A^8
        }
    }
    /* A^48 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[40 + i + nrow * j] = bigDotProd(JUMP_MATRIX_R + 5 * 64 + i * 8, JUMP_MATRIX_8s_64 + j * nrow + 8); // A^32 * A^16
        }
    }
    /* A^56 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[48 + i + nrow * j] = bigDotProd(JUMP_MATRIX_R + 5 * 64 + i * 8, JUMP_MATRIX_8s_64 + j * nrow + 16); // A^32 * A^24
        }
    }
    /* A^64 */
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            JUMP_MATRIX_8s_64[56 + i + nrow * j] = bigDotProd(JUMP_MATRIX_R + 5 * 64 + i * 8, JUMP_MATRIX_8s_64 + j * nrow + 24); // A^32 * A^32
        }
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Jumping the state ahead by a 200-bit value with the zero index as the LSB
void mrg8::jump_ahead(const short jump_val_bin[200], uint32_t *new_state)
{
	uint32_t jump_mat[8][8], rowVec[8];

	//calculating the jump matrix
	jump_calc(jump_val_bin, jump_mat);

	// Multiply the current state by jump_mat
	for (int r = 0; r < 8; ++r) {
		for (int c = 0; c < 8; ++c) {
			rowVec[c] = jump_mat[r][c];
		}
		new_state[r] = bigDotProd(rowVec, state);
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Jumping the state ahead by a jum_val
void mrg8::jump_ahead(const uint64_t jump_val, uint32_t *new_state)
{
     short jump_val_bin[200];
     dec2bin(jump_val, jump_val_bin);
     jump_ahead(jump_val_bin, new_state);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::dec2bin(const uint64_t jval, short jump_val_bin[200]) const
{
	for (int nb = 0; nb < 200; ++nb) {
		jump_val_bin[nb] = 0;
	}

	for (int nb = 0; nb < 64; ++nb) {
		if (jval & (1ul << nb))
			jump_val_bin[nb] = 1;
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Calculating the jump matrix for a given 200-bit jump value
// jump_val_bin is the jump value  with jump_val_bin[0] as the LSB
void mrg8::jump_calc(const short jump_val_bin[200], uint32_t jump_mat[8][8])
{
	uint32_t tmp_mat[8][8], vec1[8], vec2[8];

	for (int r = 0; r < 8; ++r) {
		for (int c = 0; c < 8; ++c) {
			if (r == c)
				jump_mat[r][c] = 1;
			else
				jump_mat[r][c] = 0;
		}
	}

	for (int nb = 0; nb < 200; ++nb) {
		if (jump_val_bin[nb]) {
			for (int r = 0; r < 8; ++r) {
				for (int c = 0; c < 8; ++c) {
					for (int q = 0; q < 8; ++q) {
						vec1[q] = jump_mat[r][q];
						vec2[q] = JUMP_MATRIX[64 * nb + q + 8 * c];
					}
					tmp_mat[r][c] = bigDotProd(vec1, vec2);
				}
			}

			for (int r = 0; r < 8; ++r)
				for (int c = 0; c < 8; ++c)
					jump_mat[r][c] = tmp_mat[r][c];

		}
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// A multiplicative congruential model to initialize the state using a 32-bit integer
void mrg8::mcg64ni()
{
	uint64_t x, tmp;
	uint64_t ia = 6364136223846793005ull;

	if(iseed == 0)
		iseed = 97531;

	x = iseed;
	for (int k = 0; k < 8; ++k) {
		x = ia * x;
		tmp = (x >> 32);
		state[k] = (tmp>>1);
	}
    cudaMalloc((void **)&d_state, sizeof(uint32_t) * 8);
    uint32_t tmp_state[8];
    for (int i = 0; i < 8; ++i) {
        tmp_state[i] = state[7 - i];
        // printf("d_state[%d] = %d\n", i, tmp_state[i]);
    }
    cudaMemcpy(d_state, tmp_state, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Generate integer uniform RVs in [0, 2^31 -1)
void mrg8::randint(uint32_t * iran, int n, uint32_t *new_state)
{
	uint32_t a[8], z;
	a[0] = COEFF0;
	a[1] = COEFF1;
	a[2] = COEFF2;
	a[3] = COEFF3;
	a[4] = COEFF4;
	a[5] = COEFF5;
	a[6] = COEFF6;
	a[7] = COEFF7;

    for (int k = 0; k < n; ++k) {
        z = bigDotProd(a, new_state);
        
        new_state[7] = new_state[6];// S[n-8] = S[n-7]
        new_state[6] = new_state[5];// S[n-7] = S[n-6]
        new_state[5] = new_state[4];// S[n-6] = S[n-5]
        new_state[4] = new_state[3];// S[n-5] = S[n-4]
        new_state[3] = new_state[2];// S[n-4] = S[n-3]
        new_state[2] = new_state[1];// S[n-3] = S[n-2]
        new_state[1] = new_state[0];// S[n-2] = S[n-1]
        new_state[0] = z;
        iran[k] = new_state[0];// y[n] = S[n]
	}
}

void mrg8::randint(uint32_t * iran, int n)
{
    uint32_t *new_state = new uint32_t[8];
    for (int i = 0; i < 8; ++i) {
        new_state[i] = state[i];
    }
    randint(iran, n, new_state);
    delete[] new_state;
}

uint32_t mrg8::randint()
{
	uint32_t r[1];
	randint(r, 1);
	return r[0];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Generate uniform RVs in [0, 1)
void mrg8::rand(double * fran, int n, uint32_t *new_state)
{
	uint32_t * iran1 = new uint32_t[n];
	double rnorm = 1.0/static_cast<double>(MASK);
	randint(iran1, n, new_state);
	for (int i = 0; i < n; ++i) {
	   fran[i] = static_cast<double>(iran1[i]) * rnorm;
	}
	delete [] iran1;
}

void mrg8::rand(double * ran, int n)
{
    uint32_t *new_state = new uint32_t[8];
    for (int i = 0; i < 8; ++i) {
        new_state[i] = state[i];
    }
    rand(ran, n, new_state);
    delete[] new_state;
}

double mrg8::rand()
{
	double r[1];
	rand(r, 1);
	return r[0];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
__global__ void rng_inner(double *d_ran, const uint32_t* __restrict__ d_JUMP_MATRIX, const uint32_t* __restrict__ d_state,
                          const int n, const uint64_t MASK, const double rnorm, const int each_itr)
{
    int mat_id = blockIdx.x * each_itr;
    int rid = threadIdx.x / 8;
    int cid = threadIdx.x % 8;
    uint64_t jump_val = mat_id;
    uint64_t s;
    __shared__ uint32_t d_new_state[8];
    int max_itr = (blockIdx.x < gridDim.x - 1)? each_itr : n - each_itr * blockIdx.x;
    
    if (cid == 0) {
        d_new_state[rid] = d_state[rid];
    }
    __syncthreads();
    
    /* Compute new_state by 64 threads */
    for (int nb = 0; nb < 32; ++nb) {
        if (jump_val & (1ul << nb)) {
            s = (uint64_t)(d_JUMP_MATRIX[nb * 64 + rid * 8 + cid]) * (uint64_t)(d_new_state[cid]);
            s += __shfl_xor(s, 4);
            s += __shfl_xor(s, 2);
            s = (s & MASK) + (s >> 31);
            s += __shfl_xor(s, 1);
            s = (s & MASK) + (s >> 31);
            __syncthreads();
            if (cid == 0) {
                d_new_state[rid] = s;
            }
        }
        __syncthreads();
    }
    
    
    /* Compute 8 random numbers */
    int itr;
    for (itr = 0; itr < max_itr - 8; itr += 8) {
        s = (uint64_t)(d_JUMP_MATRIX[64 * 3 + rid * 8 + cid]) * (uint64_t)(d_new_state[cid]);
        s += __shfl_xor(s, 4);
        s += __shfl_xor(s, 2);
        s = (s & MASK) + (s >> 31);
        s += __shfl_xor(s, 1);
        s = (s & MASK) + (s >> 31);
    
        __syncthreads();
        if (cid == 0) {
            d_new_state[rid] = s;
            d_ran[mat_id + itr + rid] = (double)s * rnorm;
        }
        __syncthreads();
    }
    
    for (; itr < max_itr; itr += 8) {
        s = (uint64_t)(d_JUMP_MATRIX[64 * 3 + rid * 8 + cid]) * (uint64_t)(d_new_state[cid]);
        s += __shfl_xor(s, 4);
        s += __shfl_xor(s, 2);
        s = (s & MASK) + (s >> 31);
        s += __shfl_xor(s, 1);
        s = (s & MASK) + (s >> 31);
    
        __syncthreads();
        if (cid == 0 && mat_id + itr + rid < n) {
            d_new_state[rid] = s;
            d_ran[mat_id + itr + rid] = (double)s * rnorm;
        }
        __syncthreads();
    }
}

__global__ void rng_outer_32(double *d_ran,
                             const uint32_t* __restrict__ d_JUMP_MATRIX,
                             const uint32_t* __restrict__ d_JUMP_MATRIX_8s_32,
                             const uint32_t* __restrict__ d_state,
                             const int n, const uint64_t MASK, const double rnorm, const int each_itr)
{
    const int warp_id = threadIdx.x >> 5;
    const int warp_num = blockDim.x >> 5;
    const int mat_id = (blockIdx.x * warp_num + warp_id) * each_itr;
    const int rid = threadIdx.x & 31;
    const int lrid = rid >> 3;
    const int lcid = threadIdx.x & 7;
    
    const uint64_t jump_val = mat_id;
    const int tb_offset = warp_id << 3;
    uint64_t s, s1, s2;
    __shared__ uint32_t d_new_state[32 * 8];
    
    int max_itr = each_itr;
    if (blockIdx.x == gridDim.x - 1 && warp_id == 1) {
        max_itr = n - each_itr * (blockIdx.x * warp_num + warp_id);
    }

    if (rid < 8) {
        d_new_state[tb_offset + rid] = d_state[rid];
    }
    
    /* Compute new_state by 32 threads */
    for (int nb = 0; nb < 32; ++nb) {
        if (jump_val & (1ul << nb)) {
            s1 = (uint64_t)(d_JUMP_MATRIX[(nb << 6) + (lrid << 3) + lcid]) * (uint64_t)(d_new_state[tb_offset + lcid]);
            s1 += __shfl_xor(s1, 4);
            s1 += __shfl_xor(s1, 2);
            s1 = (s1 & MASK) + (s1 >> 31);
            s1 += __shfl_xor(s1, 1);
            s1 = (s1 & MASK) + (s1 >> 31);
            
            s2 = (uint64_t)(d_JUMP_MATRIX[(nb << 6) + ((lrid + 4) << 3) + lcid]) * (uint64_t)(d_new_state[tb_offset + lcid]);
            s2 += __shfl_xor(s2, 4);
            s2 += __shfl_xor(s2, 2);
            s2 = (s2 & MASK) + (s2 >> 31);
            s2 += __shfl_xor(s2, 1);
            s2 = (s2 & MASK) + (s2 >> 31);

            if (lcid == 0) {
                d_new_state[tb_offset + lrid] = s1;
                d_new_state[tb_offset + lrid + 4] = s2;
            }
        }
    }
    
    int itr;
    for (itr = 0; itr < max_itr - 32; itr += 32) {
        s1 = 0;
        s2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s1 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + (i << 5)]) * (uint64_t)(d_new_state[tb_offset + i]);
            s2 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)]) * (uint64_t)(d_new_state[tb_offset + i + 4]);
        }
        s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
        s = (s & MASK) + (s >> 31);
        if (rid >= 24) {
            d_new_state[tb_offset + rid - 24] = s;
        }
        d_ran[mat_id + itr + rid] = (double)s * rnorm;
    }
    
    for (; itr < max_itr; itr += 32) {
        s1 = 0;
        s2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s1 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + 32 * i]) * (uint64_t)(d_new_state[tb_offset + i]);
            s2 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + 32 * (i + 4)]) * (uint64_t)(d_new_state[tb_offset + i + 4]);
        }
        s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
        s = (s & MASK) + (s >> 31);
        if (rid >= 24) {
            d_new_state[tb_offset + rid - 24] = s;
        }
        if (mat_id + itr + rid < n) {
            d_ran[mat_id + itr + rid] = (double)s * rnorm;
        }
    }
}

__global__ void rng_outer_64(double *d_ran,
                             const uint32_t* __restrict__ d_JUMP_MATRIX,
                             const uint32_t* __restrict__ d_JUMP_MATRIX_8s_64,
                             const uint32_t* __restrict__ d_state,
                             const int n, const uint64_t MASK, const double rnorm, const int each_itr)
{
    const int mat_id = blockIdx.x * each_itr;
    const int rid = threadIdx.x;
    const int lrid = threadIdx.x / 8;
    const int lcid = threadIdx.x & 7;
    
    const uint64_t jump_val = mat_id;
    const int nrow = 64;
    uint64_t s, s1, s2;
    __shared__ uint32_t d_new_state[2][8];
    
    int max_itr = each_itr;
    if (blockIdx.x == gridDim.x - 1) {
        max_itr = n - each_itr * blockIdx.x;
    }

    if (rid < 8) {
        d_new_state[0][rid] = d_state[rid];
    }
    __syncthreads();
    
    /* Compute new_state by 32 threads */
    int target = 0;
    for (int nb = 0; nb < 32; ++nb) {
        if (jump_val & (1ul << nb)) {
            s = (uint64_t)(d_JUMP_MATRIX[nb * 64 + lrid * 8 + lcid]) * (uint64_t)(d_new_state[target][lcid]);
            s += __shfl_xor(s, 4);
            s += __shfl_xor(s, 2);
            s = (s & MASK) + (s >> 31);
            s += __shfl_xor(s, 1);
            s = (s & MASK) + (s >> 31);
            target = 1 - target;
            if (lcid == 0) {
                d_new_state[target][lrid] = s;
            }
        }
        __syncthreads();
    }
    
    int itr;
    for (itr = 0; itr < max_itr - nrow; itr += nrow) {
        s1 = 0;
        s2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s1 += (uint64_t)(d_JUMP_MATRIX_8s_64[rid + (i * nrow)]) * (uint64_t)(d_new_state[target][i]);
            s2 += (uint64_t)(d_JUMP_MATRIX_8s_64[rid + ((i + 4) * nrow)]) * (uint64_t)(d_new_state[target][i + 4]);
        }
        s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
        s = (s & MASK) + (s >> 31);
        target = 1 - target;
        if (rid >= nrow - 8) {
            d_new_state[target][rid - (nrow - 8)] = s;
        }
        d_ran[mat_id + itr + rid] = (double)s * rnorm;
        __syncthreads();
    }
    
    for (; itr < max_itr; itr += nrow) {
        s1 = 0;
        s2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s1 += (uint64_t)(d_JUMP_MATRIX_8s_64[rid + nrow * i]) * (uint64_t)(d_new_state[target][i]);
            s2 += (uint64_t)(d_JUMP_MATRIX_8s_64[rid + nrow * (i + 4)]) * (uint64_t)(d_new_state[target][i + 4]);
        }
        s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
        s = (s & MASK) + (s >> 31);
        target = 1 - target;
        if (rid >= nrow - 8) {
            d_new_state[target][rid - (rid - 8)] = s;
        }
        if (mat_id + itr + rid < n) {
            d_ran[mat_id + itr + rid] = (double)s * rnorm;
        }
        __syncthreads();
    }
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::mrg8_inner(double *d_ran, const int n, const int TNUM)
{
    int pnum = TNUM / 64;
    int each_itr = n / pnum;
    rng_inner<<<pnum, 64>>>(d_ran, d_JUMP_MATRIX, d_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();

}

void mrg8::mrg8_outer_32(double *d_ran, const int n, const int TNUM)
{
    int pnum = TNUM / 32;
    int each_itr = n / pnum;

    int BS = 32;
    pnum /= (BS / 32);
    rng_outer_32<<<pnum, BS>>>(d_ran, d_JUMP_MATRIX, d_JUMP_MATRIX_8s_32, d_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();
}

void mrg8::mrg8_outer_32(double *d_ran, const int n, const int BS, const int TNUM)
{
    int pnum = TNUM / 32;
    int each_itr = n / pnum;

    pnum /= (BS / 32);
    rng_outer_32<<<pnum, BS>>>(d_ran, d_JUMP_MATRIX, d_JUMP_MATRIX_8s_32, d_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();
}

void mrg8::mrg8_outer_64(double *d_ran, const int n, const int TNUM)
{
    int pnum = TNUM / 64;
    int each_itr = n / pnum;

    int BS = 64;
    rng_outer_64<<<pnum, BS>>>(d_ran, d_JUMP_MATRIX, d_JUMP_MATRIX_8s_64, d_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::set_state(const uint32_t st[8]){
	 for (int i=0; i<8;++i)
    	 state[i] = st[i] ;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initializing the state with a 32-bit seed
void mrg8::seed_init(const uint32_t seed_val){
	iseed = seed_val;
	mcg64ni();
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::get_state(uint32_t st[8]) const{
	for (int k=0;k<8;++k)
		st[k] = state[k];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::print_state() const{
	std::cout << "(S[n-1], S[n-2], ..., S[n-8])= (";
     for (int i=0; i<7;++i)
    	 std::cout<<std::setw(10)<<state[i]<<", ";
     std::cout << state[7]<<")"<<std::endl;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::print_matrix(const uint32_t jm[8][8]) const{
	for (int i=0; i<8;++i){
		for(int j=0;j<8;++j)
			std::cout<<std::setw(10)<<jm[i][j]<<" ";
		std::cout << std::endl;
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
