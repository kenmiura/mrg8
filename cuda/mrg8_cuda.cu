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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8_cuda::mrg8_cuda(): mrg8()
{
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
mrg8_cuda::mrg8_cuda(const uint32_t seed_val): mrg8(seed_val)
{
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

// initializing the state with a 32-bit seed
void mrg8_cuda::seed_init(const uint32_t seed_val)
{
	iseed = seed_val;
	mcg64ni();
}

void mrg8_cuda::state_cpy_DtH()
{
    uint32_t tmp_state[8];
    cudaMemcpy(tmp_state, d_state, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        state[i] = tmp_state[7 - i];
    }
}

void mrg8_cuda::reverse_jump_matrix()
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

void mrg8_cuda::set_jump_matrix_8s_32()
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

void mrg8_cuda::set_jump_matrix_8s_64()
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
// A multiplicative congruential model to initialize the state using a 32-bit integer
void mrg8_cuda::mcg64ni()
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
    }
    cudaMemcpy(d_state, tmp_state, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
__global__ void rng_inner(double *d_ran, const uint32_t* __restrict__ d_JUMP_MATRIX,
                          const uint32_t* __restrict__ d_state, uint32_t *d_next_state,
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
            s--;
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
        if (cid == 0 && itr + rid < max_itr) {
            d_new_state[rid] = s;
            s--;
            d_ran[mat_id + itr + rid] = (double)s * rnorm;
        }
        __syncthreads();
    }

    /* Store next state */
    if (blockIdx.x == gridDim.x - 1 && cid == 0) {
        int perm = itr - max_itr;
        d_next_state[(rid + perm) % 8] = d_new_state[rid];
    }
}

__global__ void rng_outer_32(double *d_ran,
                             const uint32_t* __restrict__ d_JUMP_MATRIX,
                             const uint32_t* __restrict__ d_JUMP_MATRIX_8s_32,
                             const uint32_t* __restrict__ d_state, uint32_t *d_next_state,
                             const int n, const uint64_t MASK, const double rnorm, const int each_itr)
{
    const int warp_id = threadIdx.x >> 5;
    const int warp_num = blockDim.x >> 5;
    const int mat_id = (blockIdx.x * warp_num + warp_id) * each_itr;
    const int rid = threadIdx.x & 31;
    const int lrid = rid >> 3;
    const int lcid = threadIdx.x & 7;
    const int sid = (rid + 8) & 31;
    
    const uint64_t jump_val = mat_id;
    const int tb_offset = warp_id << 5;
    uint64_t s, s1, s2;
    __shared__ uint32_t d_new_state[32 * 32];
    // __shared__ uint32_t shared_JUMP_MATRIX_8s_32[64 * 4];
    int max_itr = each_itr;
    if (blockIdx.x == gridDim.x - 1 && warp_id == warp_num - 1) {
        max_itr = n - each_itr * (blockIdx.x * warp_num + warp_id);
    }

    if (rid < 8) {
        d_new_state[tb_offset + rid] = d_state[rid];
    }
    
    // if (threadIdx.x < 32) {
    //     for (int i = 0; i < 8; ++i) {
    //         shared_JUMP_MATRIX_8s_32[threadIdx.x + i * 32] = d_JUMP_MATRIX_8s_32[threadIdx.x + i * 32];
    //     }
    // }
    // __syncthreads();

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
    uint64_t l1;
    uint32_t u1, u2;
    for (itr = 0; itr < max_itr - 32; itr += 32) {
        l1 = 0;
        u1 = 0;
        u2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            /* Compute full bits */
            // s1 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + (i << 5)]) * (uint64_t)(d_new_state[tb_offset + i]);
            // s2 += (uint64_t)(d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)]) * (uint64_t)(d_new_state[tb_offset + i + 4]);

            /* Compute lower 32-bit */
            l1 += (uint64_t)((d_JUMP_MATRIX_8s_32[rid + (i << 5)]) * (d_new_state[tb_offset + i]));
            l1 += (uint64_t)((d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)]) * (d_new_state[tb_offset + i + 4]));
            /* Compute upper 32-bit */
            u1 += __umulhi(d_JUMP_MATRIX_8s_32[rid + (i << 5)], d_new_state[tb_offset + i]);
            u2 += __umulhi(d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)], d_new_state[tb_offset + i + 4]);
        }
        
        s = (((uint64_t)u1 + (uint64_t)u2) << 1) + (l1 & MASK) + (l1 >> 31);
        
        s = (s & MASK) + (s >> 31);

        d_new_state[tb_offset + sid] = s;
        
        d_ran[mat_id + itr + rid] = (double)(s - 1) * rnorm;
    }

    if (itr + rid < max_itr) {
        l1 = 0;
        u1 = 0;
        u2 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // compute lower 32-bit
            l1 = l1 + (uint64_t)((d_JUMP_MATRIX_8s_32[rid + (i << 5)]) * (d_new_state[tb_offset + i]));
            l1 = l1 + (uint64_t)((d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)]) * (d_new_state[tb_offset + i + 4]));
            // compute upper 32-bit
            u1 = u1 + __umulhi(d_JUMP_MATRIX_8s_32[rid + (i << 5)], d_new_state[tb_offset + i]);
            u2 = u2 + __umulhi(d_JUMP_MATRIX_8s_32[rid + ((i + 4) << 5)], d_new_state[tb_offset + i + 4]);
        }
        
        s = (((uint64_t)u1 + (uint64_t)u2) << 1) + (l1 & MASK) + (l1 >> 31);

        s = (s & MASK) + (s >> 31);
        if (itr + rid >= max_itr - 8) {
            d_new_state[tb_offset + (rid % 8)] = s;
        }
        // d_new_state[tb_offset + sid] = s;
        if (mat_id + itr + rid < n) {
            d_ran[mat_id + itr + rid] = (double)(s - 1) * rnorm;
        }
    }
    if (blockIdx.x == gridDim.x - 1 && warp_id == warp_num - 1) {
        if (rid < 8) {
            int offset = (max_itr - itr) % 8;
            d_next_state[rid] = d_new_state[tb_offset + ((rid + offset) % 8)];
        }
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8_cuda::mrg8_inner(double *d_ran, const int n, const int TNUM)
{
    int pnum = TNUM / 64;
    int each_itr = n / pnum;
    uint32_t *d_next_state;
    cudaMalloc((void **)&d_next_state, sizeof(uint32_t) * 8);
    rng_inner<<<pnum, 64>>>(d_ran, d_JUMP_MATRIX, d_state, d_next_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();
    cudaMemcpy(d_state, d_next_state, sizeof(uint32_t) * 8, cudaMemcpyDeviceToDevice);
    cudaFree(d_next_state);
}

void mrg8_cuda::mrg8_outer_32(double *d_ran, const int n, const int BS, const int TNUM)
{
    int pnum = TNUM / 32;
    int each_itr = n / pnum;
    uint32_t *d_next_state;
    cudaMalloc((void **)&d_next_state, sizeof(uint32_t) * 8);

    pnum /= (BS / 32);
    rng_outer_32<<<pnum, BS>>>(d_ran, d_JUMP_MATRIX, d_JUMP_MATRIX_8s_32, d_state, d_next_state, n, MASK, 1.0 / static_cast<double>(MASK), each_itr);
    cudaThreadSynchronize();
    cudaMemcpy(d_state, d_next_state, sizeof(uint32_t) * 8, cudaMemcpyDeviceToDevice);
    cudaFree(d_next_state);
}

void mrg8_cuda::mrg8_outer_32(double *d_ran, const int n, const int TNUM)
{
    mrg8_outer_32(d_ran, n, 64, TNUM);
}

void mrg8_cuda::mrg8_outer_32(double *d_ran, const int n)
{
    mrg8_outer_32(d_ran, n, 64 * 1024);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
