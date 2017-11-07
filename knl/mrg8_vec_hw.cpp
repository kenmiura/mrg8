/*
 * mrg8.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: Oct 31, 2017
 *      Author: Yusuke
 */

#include "mrg8_vec.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <ctime>
#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <zmmintrin.h>

using namespace std;

#define AVX2

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8_vec::mrg8_vec(): mrg8()
{
    read_jump_matrix();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_IP_MATRIX[(7 - i) * 8 + (7 - j)] = (uint64_t)(JUMP_MATRIX[8 * 8 * 2 + i + j * 8]);
        }
    }
    for (int i = 0; i < 64; ++i) {
        A8_OP_MATRIX[i] = (uint64_t)(JUMP_MATRIX[8 * 8 * 3 - 1 - i]);
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8_vec::mrg8_vec(const uint32_t seed_val): mrg8(seed_val)
{
    read_jump_matrix();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_IP_MATRIX[(7 - i) * 8 + (7 - j)] = (uint64_t)(JUMP_MATRIX[8 * 8 * 2 + i + j * 8]);
        }
    }
    for (int i = 0; i < 64; ++i) {
        A8_OP_MATRIX[i] = (uint64_t)(JUMP_MATRIX[8 * 8 * 3 - 1 - i]);
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8_vec::mrg8_vec_inner(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX2
    int i, j, k;
    double rnorm = 1.0 / static_cast<double>(MASK);
    const __m256i mask_m = _mm256_set1_epi64x(MASK);
    const __m256d rnorm_m = _mm256_set1_pd(rnorm);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
    int64_t r_state[8];
    __m256i state1_m[2], state2_m[2], s1_m, s2_m, s_m, a_m;
    __m256d ran_m;
    __m256i s_32m;
    __m256i mone_m = _mm256_set1_epi64x(-1);
    uint64_t s;
    
    for (i = 0; i < 8; ++i) {
        r_state[i] = (uint64_t)(each_state[7 - i]);
    }

    state1_m[0] = _mm256_maskload_epi64(r_state, true_m);
    state2_m[0] = _mm256_maskload_epi64(r_state + 4, true_m);
    int target;
    for (i = 0; i < n - 4; i+=4) {
        target = (i >> 2) & 1;
        for (k = 0; k < 4; ++k) {
            a_m = _mm256_maskload_epi64(A8_IP_MATRIX + (k + 4) * 8, true_m);
            s1_m = _mm256_mul_epu32(a_m, state1_m[target]);
            a_m = _mm256_maskload_epi64(A8_IP_MATRIX + (k + 4) * 8 + 4, true_m);
            s2_m = _mm256_mul_epu32(a_m, state2_m[target]);
            s_m = _mm256_add_epi64(s1_m, s2_m);
            
            s1_m = _mm256_and_si256(s_m, mask_m);
            s2_m = _mm256_srli_epi64(s_m, 31);
            s_m = _mm256_add_epi64(s1_m, s2_m);
            s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

            state1_m[1 - target][k] = state2_m[target][k];
            state2_m[1 - target][k] = s;
        }
        s_m = _mm256_and_si256(state2_m[1 - target], mask_m);
        state2_m[1 - target] = _mm256_srli_epi64(state2_m[1 - target], 31);
        state2_m[1 - target] = _mm256_add_epi64(s_m, state2_m[1 - target]);
        s_m = _mm256_add_epi64(state2_m[1 - target], mone_m);

        ran_m = _mm256_set_pd((double)s_m[3], (double)s_m[2], (double)s_m[1], (double)s_m[0]);
        ran_m = _mm256_mul_pd(ran_m, rnorm_m);
        _mm256_store_pd(ran + i, ran_m);
    }

    /* Fraction */
    target = (i >> 2) & 1;
    for (k = 0; k < 4; ++k) {
        a_m = _mm256_maskload_epi64(A8_IP_MATRIX + (k + 4) * 8, true_m);
        s1_m = _mm256_mul_epu32(a_m, state1_m[target]);
        a_m = _mm256_maskload_epi64(A8_IP_MATRIX + (k + 4) * 8 + 4, true_m);
        s2_m = _mm256_mul_epu32(a_m, state2_m[target]);
        s_m = _mm256_add_epi64(s1_m, s2_m);
            
        s1_m = _mm256_and_si256(s_m, mask_m);
        s2_m = _mm256_srli_epi64(s_m, 31);
        s_m = _mm256_add_epi64(s1_m, s2_m);
        s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

        state1_m[1 - target][k] = state2_m[target][k];
        state2_m[1 - target][k] = s;
    }
    s_m = _mm256_and_si256(state2_m[1 - target], mask_m);
    state2_m[1 - target] = _mm256_srli_epi64(state2_m[1 - target], 31);
    state2_m[1 - target] = _mm256_add_epi64(s_m, state2_m[1 - target]);

    s_m = _mm256_add_epi64(state2_m[1 - target], mone_m);
    ran_m = _mm256_set_pd((double)s_m[3], (double)s_m[2], (double)s_m[1], (double)s_m[0]);

    ran_m = _mm256_mul_pd(ran_m, rnorm_m);
    for (k = 0; k < n - i; ++k) {
        ran[i + k] = ran_m[k];
    }
    
    for (i = 0; i < k; ++i) {
        each_state[k - 1 - i] = state2_m[1 - target][i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[k + i] = state2_m[target][3 - i];
    }
    for (i = 0; i < k; ++i) {
        each_state[4 + k - 1 - i] = state1_m[1 - target][i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[4 + k + i] = state1_m[target][3 - i];
    }

#else
    int i, j, k;
    uint32_t r_state[2][8];
    uint64_t s1, s2, s;
    double rnorm = 1.0 / static_cast<double>(MASK);
    int target;
    for (i = 0; i < 8; ++i) {
        r_state[0][i] = each_state[7 - i];
    }
    
    for (i = 0; i < n; i+=4) {
        target = (i >> 2) & 1;
        for (k = 0; k < 4 && i + k < n; ++k) {
            s1 = 0;
            s2 = 0;
            for (j = 0; j < 4; ++j) {
                s1 += (uint64_t)(A8_IP_MATRIX[(4 + k) * 8 + j]) * r_state[target][j];
                s2 += (uint64_t)(A8_IP_MATRIX[(4 + k) * 8 + j + 4]) * r_state[target][j + 4];
            }
            s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
            r_state[1 - target][k] = r_state[target][4 + k];
            r_state[1 - target][4 + k] = (s & MASK) + (s >> 31);
            ran[i + k] = static_cast<double>(r_state[1 - target][4 + k] - 1) * rnorm;
        }
    }
    for (i = 0; i < k; ++i) {
        each_state[k - 1 - i] = r_state[1 - target][4 + i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[k + i] = r_state[target][7 - i];
    }
    for (i = 0; i < k; ++i) {
        each_state[4 + k - 1 - i] = r_state[1 - target][i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[4 + k + i] = r_state[target][3 - i];
    }
#endif
}

void mrg8_vec::mrg8_vec_inner(double * ran, int n)
{
    mrg8_vec_inner(ran, n, state);
}

double mrg8_vec::mrg8_vec_inner()
{
    double r;
    mrg8_vec_inner(&r, 1, state);
    return r;
}

double mrg8_vec::mrg8_vec_inner(uint32_t *new_state)
{
    double r;
    mrg8_vec_inner(&r, 1, new_state);
    return r;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8_vec::mrg8_vec_outer(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX2
    int i, j;
    int64_t r_state[8];
    __m256i a_m[2], s_m, s1_m, s2_m, state_m[2];
    __m256d ran_m;
    double rnorm = 1.0 / static_cast<double>(MASK);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
    const __m256i mask_m = _mm256_set1_epi64x(MASK);
    const __m256i mone_m = _mm256_set1_epi64x(-1);
    const __m256d rnorm_m = _mm256_set1_pd(rnorm);

    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
    }
    state_m[0] = _mm256_maskload_epi64(r_state, true_m);
    state_m[1] = _mm256_maskload_epi64(r_state + 4, true_m);

    for (i = 0; i < n - 4; i+=4) {
        s1_m = _mm256_set1_epi64x(0);
        s2_m = _mm256_set1_epi64x(0);

        for (j = 0; j < 4; ++j) {
            a_m[0] = _mm256_maskload_epi64(A8_OP_MATRIX + j * 8 + 4, true_m);
            a_m[1] = _mm256_maskload_epi64(A8_OP_MATRIX + (j + 4) * 8 + 4, true_m);
            s_m = _mm256_set1_epi64x((uint64_t)(state_m[0][j]));
            s_m = _mm256_mul_epu32(a_m[0], s_m);
            s1_m = _mm256_add_epi64(s1_m, s_m);

            s_m = _mm256_set1_epi64x((uint64_t)(state_m[1][j]));
            s_m = _mm256_mul_epu32(a_m[1], s_m);
            s2_m = _mm256_add_epi64(s2_m, s_m);
        }

        s_m = _mm256_and_si256(s1_m, mask_m);
        s1_m = _mm256_srli_epi64(s1_m, 31);
        s1_m = _mm256_add_epi64(s_m, s1_m);

        s_m = _mm256_and_si256(s2_m, mask_m);
        s2_m = _mm256_srli_epi64(s2_m, 31);
        s2_m = _mm256_add_epi64(s_m, s2_m);
        
        s_m = _mm256_add_epi64(s1_m, s2_m);
            
        state_m[0] = state_m[1];

        state_m[1] = _mm256_and_si256(s_m, mask_m);
        s_m = _mm256_srli_epi64(s_m, 31);
        state_m[1] = _mm256_add_epi64(s_m, state_m[1]);
        s_m = _mm256_add_epi64(state_m[1], mone_m);
        
        ran_m = _mm256_set_pd((double)s_m[3], (double)s_m[2], (double)s_m[1], (double)s_m[0]);

        ran_m = _mm256_mul_pd(ran_m, rnorm_m);
        _mm256_store_pd(ran + i, ran_m);
    }

    /* Fraction */
    s1_m = _mm256_set1_epi64x(0);
    s2_m = _mm256_set1_epi64x(0);
    for (j = 0; j < 4; ++j) {
        a_m[0] = _mm256_maskload_epi64(A8_OP_MATRIX + j * 8 + 4, true_m);
        a_m[1] = _mm256_maskload_epi64(A8_OP_MATRIX + (j + 4) * 8 + 4, true_m);

        r_state[j] = state_m[0][j];
        s_m = _mm256_set1_epi64x((uint64_t)(state_m[0][j]));
        s_m = _mm256_mul_epu32(a_m[0], s_m);
        s1_m = _mm256_add_epi64(s1_m, s_m);

        r_state[4 + j] = state_m[1][j];
        s_m = _mm256_set1_epi64x((uint64_t)(state_m[1][j]));
        s_m = _mm256_mul_epu32(a_m[1], s_m);
        s2_m = _mm256_add_epi64(s2_m, s_m);
    }

    s_m = _mm256_and_si256(s1_m, mask_m);
    s1_m = _mm256_srli_epi64(s1_m, 31);
    s1_m = _mm256_add_epi64(s_m, s1_m);

    s_m = _mm256_and_si256(s2_m, mask_m);
    s2_m = _mm256_srli_epi64(s2_m, 31);
    s2_m = _mm256_add_epi64(s_m, s2_m);
        
    s_m = _mm256_add_epi64(s1_m, s2_m);
            
    state_m[0] = state_m[1];

    state_m[1] = _mm256_and_si256(s_m, mask_m);
    s_m = _mm256_srli_epi64(s_m, 31);
    state_m[1] = _mm256_add_epi64(s_m, state_m[1]);
    s_m = _mm256_add_epi64(state_m[1], mone_m);
    
    ran_m = _mm256_set_pd((double)s_m[3], (double)s_m[2], (double)s_m[1], (double)s_m[0]);
    
    ran_m = _mm256_mul_pd(ran_m, rnorm_m);
    for (j = 0; j < n - i; ++j) {
        ran[i + j] = ran_m[j];
    }

    for (i = 0; i < j; ++i) {
        each_state[j - 1 - i] = state_m[1][i];
    }
    for (i = 0; i < 4 - j; ++i) {
        each_state[j + i] = r_state[7 - i];
    }
    for (i = 0; i < j; ++i) {
        each_state[4 + j - 1 - i] = state_m[0][i];
    }
    for (i = 0; i < 4 - j; ++i) {
        each_state[4 + j + i] = r_state[3 - i];
    }
#else
    int i, j, k;
    uint32_t r_state[8];
    uint64_t s1[8], s2[8], s[8];
    double rnorm = 1.0 / static_cast<double>(MASK);

    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
    }
    
    for (i = 0; i < n; i+=4) {
#pragma simd
        for (k = 0; k < 4; ++k) {
            s1[k] = 0;
            s2[k] = 0;
        }
        for (j = 0; j < 4; ++j) {
#pragma simd
            for (k = 0; k < 4; ++k) {
                s1[k] += (uint64_t)(A8_OP_MATRIX[j * 8 + k + 4]) * r_state[j];
                s2[k] += (uint64_t)(A8_OP_MATRIX[(j + 4) * 8 + k + 4]) * r_state[j + 4];
            }
        }
        for (k = 0; k < 4 && i + k < n; ++k) { //only unroll not vectorized
            s[k] = (s1[k] & MASK) + (s1[k] >> 31) + (s2[k] & MASK) + (s2[k] >> 31);
            r_state[k] = r_state[4 + k];
            r_state[4 + k] = (s[k] & MASK) + (s[k] >> 31);
            ran[i + k] = static_cast<double>(r_state[4 + k] - 1) * rnorm;
        }
    }
    for (i = 0; i < k; ++i) {
        each_state[k - 1 - i] = r_state[4 + i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[k + i] = r_state[7 - i];
    }
    for (i = 0; i < k; ++i) {
        each_state[4 + k - 1 - i] = r_state[i];
    }
    for (i = 0; i < 4 - k; ++i) {
        each_state[4 + k + i] = r_state[3 - i];
    }

#endif
}
    
void mrg8_vec::mrg8_vec_outer(double * ran, const int n)
{
    mrg8_vec_outer(ran, n, state);
}
    
double mrg8_vec::mrg8_vec_outer()
{
    double r;
    mrg8_vec_outer(&r, 1, state);
    return r;
}

double mrg8_vec::mrg8_vec_outer(uint32_t *new_state)
{
    double r;
    mrg8_vec_outer(&r, 1, new_state);
    return r;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8_vec::mrg8_vec_inner_tp(double * ran, int n)
{
    int tnum = omp_get_max_threads();
    uint32_t next_state[8];
#pragma omp parallel
    {
        int each_n = n / tnum;
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        if (tid == (tnum - 1)) {
            each_n = n - each_n * tid;
        }
        jump_ahead(start, each_state);
        mrg8_vec_inner(ran + start, each_n, each_state);
        if (tid == tnum - 1) {
            for (int j = 0; j < 8; ++j) {
                next_state[j] = each_state[j];
            }
        }
        delete[] each_state;
    }
    for (int i = 0; i < 8; ++i) {
        state[i] = next_state[i];
    }
}

void mrg8_vec::mrg8_vec_outer_tp(double * ran, int n)
{
    int tnum = omp_get_max_threads();
    uint32_t next_state[8];
#pragma omp parallel
    {
        int each_n = n / tnum;
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        if (tid == (tnum - 1)) {
            each_n = n - each_n * tid;
        }
        jump_ahead(start, each_state);
        mrg8_vec_outer(ran + start, each_n, each_state);
        if (tid == tnum - 1) {
            for (int j = 0; j < 8; ++j) {
                next_state[j] = each_state[j];
            }
        }
        delete[] each_state;
    }
    for (int i = 0; i < 8; ++i) {
        state[i] = next_state[i];
    }
}
