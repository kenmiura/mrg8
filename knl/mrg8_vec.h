/*
 * mrg8.h
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: June 29, 2017
 *      Author: Yusuke
 *  Updated on: April 9, 2018
 *      Author: Yusuke
 */

#ifndef MRG8_VEC_H
#define MRG8_VEC_H

#include <vector>
#include <stdint.h>
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

#include "mrg8.h"

using namespace std;
#define AVX512

class mrg8_vec : public mrg8
{
public:

	mrg8_vec();
	mrg8_vec(const uint32_t seed_val);
    ~mrg8_vec()
    {
    }    
    void mrg8_vec_inner(double *ran, int n);
    double mrg8_vec_inner();
    double mrg8_vec_inner(uint32_t *new_state);
    void mrg8_vec_outer(double * ran, int n);
    double mrg8_vec_outer();
    double mrg8_vec_outer(uint32_t *new_state);
    double operator() ()
    {
        return mrg8_vec_outer();
    }
    
    void mrg8_vec_inner_tp(double *ran, int n);
    void mrg8_vec_outer_tp(double * ran, int n);
    void mrg8_vec_outer_tp_small(double * ran, int each_n, int it);
    void mrg8_vec_outer_tp_sub(double * ran, int n, int sub_N);

private:
    int64_t A8_IP_MATRIX[64];
    int64_t A8_OP_MATRIX[64];
    int64_t A8_OP_SH_MATRIX[64];
    uint32_t A816_OP_SH_MATRIX[128];
    void mrg8_vec_inner(double *ran, int n, uint32_t *each_state);
    void mrg8_vec_outer(double * ran, int n, uint32_t *each_state);
};


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8_vec::mrg8_vec(): mrg8()
{
    read_jump_matrix();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_IP_MATRIX[(7 - i) * 8 + (7 - j)] = (uint64_t)(JUMP_MATRIX[8 * 8 * 3 + i + j * 8]);
        }
    }
    for (int i = 0; i < 64; ++i) {
        A8_OP_MATRIX[i] = (uint64_t)(JUMP_MATRIX[8 * 8 * 4 - 1 - i]);
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_OP_SH_MATRIX[i + j * 8] = A8_OP_MATRIX[i + ((i + j) % 8) * 8];
        }
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8_vec::mrg8_vec(const uint32_t seed_val): mrg8(seed_val)
{
    read_jump_matrix();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_IP_MATRIX[(7 - i) * 8 + (7 - j)] = (uint64_t)(JUMP_MATRIX[8 * 8 * 3 + i + j * 8]);
        }
    }
    for (int i = 0; i < 64; ++i) {
        A8_OP_MATRIX[i] = (uint64_t)(JUMP_MATRIX[8 * 8 * 4 - 1 - i]);
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            A8_OP_SH_MATRIX[i + j * 8] = A8_OP_MATRIX[i + ((i + j) % 8) * 8];
        }
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
inline void mrg8_vec::mrg8_vec_inner(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX512
    int i, j, k;
    double rnorm = 1.0 / static_cast<double>(MASK);
    uint64_t r_state[8];
    __m512i mone_m = _mm512_set1_epi64(-1);
    __m512i state1_m, state2_m, s1_m, s2_m, s_m, mask_m, a_m;
    __m512d ran_m, rnorm_m;
    __m256i s_32m;
    uint64_t s;
    
    for (i = 0; i < 8; ++i) {
        r_state[i] = (uint64_t)(each_state[7 - i]);
    }

    state1_m = _mm512_load_epi64(r_state);
    mask_m = _mm512_set1_epi64(MASK);
    rnorm_m = _mm512_set1_pd(rnorm);

    for (i = 0; i < n - 8; i+=8) {
        if (((i >> 3) & 1) == 0) {
            for (k = 0; k < 8; ++k) {
                a_m = _mm512_load_epi64(A8_IP_MATRIX + k * 8);
                s1_m = _mm512_mul_epu32(a_m, state1_m);
                s_m = _mm512_and_epi64(s1_m, mask_m);
                s2_m = _mm512_srli_epi64(s1_m, 31);
                s_m = _mm512_add_epi64(s_m, s2_m);

                s = _mm512_reduce_add_epi64(s_m);
                state2_m[k] = s;
            }
            s_m = _mm512_and_epi64(state2_m, mask_m);
            state2_m = _mm512_srli_epi64(state2_m, 31);
            state2_m = _mm512_add_epi64(s_m, state2_m);
            
            s_m = _mm512_and_epi64(state2_m, mask_m);
            state2_m = _mm512_srli_epi64(state2_m, 31);
            state2_m = _mm512_add_epi64(s_m, state2_m);

            s_m = _mm512_add_epi64(state2_m, mone_m);
            s_32m = _mm512_cvtepi64_epi32(s_m);
            
            ran_m = _mm512_cvtepi32_pd(s_32m);
            ran_m = _mm512_mul_pd(ran_m, rnorm_m);
            _mm512_store_pd(ran + i, ran_m);
        }
        else {
            for (k = 0; k < 8; ++k) {
                a_m = _mm512_load_epi64(A8_IP_MATRIX + k * 8);
                s1_m = _mm512_mul_epu32(a_m, state2_m);
                s_m = _mm512_and_epi64(s1_m, mask_m);
                s2_m = _mm512_srli_epi64(s1_m, 31);
                s_m = _mm512_add_epi64(s_m, s2_m);

                s = _mm512_reduce_add_epi64(s_m);
                state1_m[k] = s;
            }
            s_m = _mm512_and_epi64(state1_m, mask_m);
            state1_m = _mm512_srli_epi64(state1_m, 31);
            state1_m = _mm512_add_epi64(s_m, state1_m);

            s_m = _mm512_and_epi64(state1_m, mask_m);
            state1_m = _mm512_srli_epi64(state1_m, 31);
            state1_m = _mm512_add_epi64(s_m, state1_m);

            s_m = _mm512_add_epi64(state1_m, mone_m);
            s_32m = _mm512_cvtepi64_epi32(s_m);

            ran_m = _mm512_cvtepi32_pd(s_32m);
            ran_m = _mm512_mul_pd(ran_m, rnorm_m);
            _mm512_store_pd(ran + i, ran_m);
        }
    }
    
    if (((i >> 3) & 1) == 0) {
        for (k = 0; k < (n - i); ++k) {
            a_m = _mm512_load_epi64(A8_IP_MATRIX + k * 8);
            s1_m = _mm512_mul_epu32(a_m, state1_m);
            s_m = _mm512_and_epi64(s1_m, mask_m);
            s2_m = _mm512_srli_epi64(s1_m, 31);
            s_m = _mm512_add_epi64(s_m, s2_m);

            s = _mm512_reduce_add_epi64(s_m);
            state2_m[k] = s;
        }
        s_m = _mm512_and_epi64(state2_m, mask_m);
        state2_m = _mm512_srli_epi64(state2_m, 31);
        state2_m = _mm512_add_epi64(s_m, state2_m);

        s_m = _mm512_and_epi64(state2_m, mask_m);
        state2_m = _mm512_srli_epi64(state2_m, 31);
        state2_m = _mm512_add_epi64(s_m, state2_m);

        s_m = _mm512_add_epi64(state2_m, mone_m);
        s_32m = _mm512_cvtepi64_epi32(s_m);
        ran_m = _mm512_cvtepi32_pd(s_32m);
        ran_m = _mm512_mul_pd(ran_m, rnorm_m);
        for (k = 0; k < n - i; ++k) {
            ran[i + k] = ran_m[k];
        }
        
        for (j = k; j < 8; ++j) {
            each_state[7 - (j - k)] = (uint32_t)(state1_m[j]);
        }
        for (j = 0; j < k; ++j) {
            each_state[k - j - 1] = (uint32_t)(state2_m[j]);
        }
    }
    else {
        for (k = 0; k < (n - i); ++k) {
            a_m = _mm512_load_epi64(A8_IP_MATRIX + k * 8);
            s1_m = _mm512_mul_epu32(a_m, state2_m);
            s_m = _mm512_and_epi64(s1_m, mask_m);
            s2_m = _mm512_srli_epi64(s1_m, 31);
            s_m = _mm512_add_epi64(s_m, s2_m);

            s = _mm512_reduce_add_epi64(s_m);
            state1_m[k] = s;
        }
        s_m = _mm512_and_epi64(state1_m, mask_m);
        state1_m = _mm512_srli_epi64(state1_m, 31);
        state1_m = _mm512_add_epi64(s_m, state1_m);

        s_m = _mm512_and_epi64(state1_m, mask_m);
        state1_m = _mm512_srli_epi64(state1_m, 31);
        state1_m = _mm512_add_epi64(s_m, state1_m);

        s_m = _mm512_add_epi64(state1_m, mone_m);
        s_32m = _mm512_cvtepi64_epi32(s_m);
        ran_m = _mm512_cvtepi32_pd(s_32m);
        ran_m = _mm512_mul_pd(ran_m, rnorm_m);
        for (k = 0; k < n - i; ++k) {
            ran[i + k] = ran_m[k];
        }
        for (j = k; j < 8; ++j) {
            each_state[7 - (j - k)] = (uint32_t)(state2_m[j]);
        }
        for (j = 0; j < k; ++j) {
            each_state[k - j - 1] = (uint32_t)(state1_m[j]);
        }
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
    
    for (i = 0; i < n; i+=8) {
        target = (i >> 3) & 1;
        for (k = 0; k < 8 && i + k < n; ++k) {
            s1 = 0;
            s2 = 0;
            for (j = 0; j < 4; ++j) {
                s1 += (uint64_t)(A8_IP_MATRIX[k * 8 + j]) * r_state[target][j];
                s2 += (uint64_t)(A8_IP_MATRIX[k * 8 + j + 4]) * r_state[target][j + 4];
            }
            s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
            s = (s & MASK) + (s >> 31);
            r_state[1 - target][k] = (s & MASK) + (s >> 31);
            ran[i + k] = static_cast<double>(r_state[1 - target][k] - 1) * rnorm;
        }
    }
    for (i = k; i < 8; ++i) {
        each_state[7 - (i - k)] = r_state[target][i];
    }
    for (i = 0; i < k; ++i) {
        each_state[k - i - 1] = r_state[1 - target][i];
    }
#endif
}

inline void mrg8_vec::mrg8_vec_inner(double * ran, int n)
{
    mrg8_vec_inner(ran, n, state);
}

inline double mrg8_vec::mrg8_vec_inner()
{
    double r;
    mrg8_vec_inner(&r, 1, state);
    return r;
}

inline double mrg8_vec::mrg8_vec_inner(uint32_t *new_state)
{
    double r;
    mrg8_vec_inner(&r, 1, new_state);
    return r;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
inline void mrg8_vec::mrg8_vec_outer(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX512
    int i, j;
    uint64_t r_state[8];
    const __m512i one_m = _mm512_set1_epi64(1);
    const __m512i idx_m = _mm512_set_epi64(0, 7, 6, 5, 4, 3, 2, 1);
    __m256i state_32m;
    __m512i state_m, s_m, s1_m, s2_m, mask_m, a_m[8];
    __m512d ran_m, rnorm_m;
    
    double rnorm = 1.0 / static_cast<double>(MASK);

    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
        a_m[i] = _mm512_load_epi64(A8_OP_SH_MATRIX + i * 8);
    }

    mask_m = _mm512_set1_epi64(MASK);
    rnorm_m = _mm512_set1_pd(rnorm);
    
    state_m = _mm512_load_epi64(r_state);

    i = 0;
    for (i = 0; i < n - 8; i+=8) {
        s1_m = _mm512_setzero_si512();
        s2_m = _mm512_setzero_si512();

        for (j = 0; j < 4; ++j) {
            s_m = _mm512_mul_epu32(a_m[j], state_m);
            s1_m = _mm512_add_epi64(s1_m, s_m);
            state_m = _mm512_permutexvar_epi64(idx_m, state_m);
        }
        for (j = 0; j < 4; ++j) {
            s_m = _mm512_mul_epu32(a_m[j + 4], state_m);
            s2_m = _mm512_add_epi64(s2_m, s_m);
            state_m = _mm512_permutexvar_epi64(idx_m, state_m);
        }
        
        s_m = _mm512_and_epi64(s1_m, mask_m);
        s1_m = _mm512_srli_epi64(s1_m, 31);
        s1_m = _mm512_add_epi64(s_m, s1_m);
        
        s_m = _mm512_and_epi64(s2_m, mask_m);
        s2_m = _mm512_srli_epi64(s2_m, 31);
        s2_m = _mm512_add_epi64(s_m, s2_m);
        
        s_m = _mm512_add_epi64(s1_m, s2_m);
        
        state_m = _mm512_and_epi64(s_m, mask_m);
        s_m = _mm512_srli_epi64(s_m, 31);
        s_m = _mm512_add_epi64(s_m, state_m);
        
        state_m = _mm512_and_epi64(s_m, mask_m);
        s_m = _mm512_srli_epi64(s_m, 31);
        state_m = _mm512_add_epi64(s_m, state_m);
        
        s_m = _mm512_sub_epi64(state_m, one_m);
        state_32m = _mm512_cvtepi64_epi32(s_m);
        
        ran_m = _mm512_cvtepi32_pd(state_32m);
        ran_m = _mm512_mul_pd(ran_m, rnorm_m);
        _mm512_store_pd(ran + i, ran_m);
    }
    
    _mm512_store_epi64(r_state, state_m);
    
    /* Fraction */
    s1_m = _mm512_setzero_si512();
    s2_m = _mm512_setzero_si512();

    for (j = 0; j < 4; ++j) {
        s_m = _mm512_mul_epu32(a_m[j], state_m);
        s1_m = _mm512_add_epi64(s1_m, s_m);
        state_m = _mm512_permutexvar_epi64(idx_m, state_m);
    }
    for (j = 0; j < 4; ++j) {
        s_m = _mm512_mul_epu32(a_m[j + 4], state_m);
        s2_m = _mm512_add_epi64(s2_m, s_m);
        state_m = _mm512_permutexvar_epi64(idx_m, state_m);
    }

    s_m = _mm512_and_epi64(s1_m, mask_m);
    s1_m = _mm512_srli_epi64(s1_m, 31);
    s1_m = _mm512_add_epi64(s_m, s1_m);

    s_m = _mm512_and_epi64(s2_m, mask_m);
    s2_m = _mm512_srli_epi64(s2_m, 31);
    s2_m = _mm512_add_epi64(s_m, s2_m);

    s_m = _mm512_add_epi64(s1_m, s2_m);
            
    state_m = _mm512_and_epi64(s_m, mask_m);
    s_m = _mm512_srli_epi64(s_m, 31);
    state_m = _mm512_add_epi64(s_m, state_m);

    s_m = _mm512_sub_epi64(state_m, one_m);
    state_32m = _mm512_cvtepi64_epi32(s_m);

    ran_m = _mm512_cvtepi32_pd(state_32m);
    ran_m = _mm512_mul_pd(ran_m, rnorm_m);
    for (j = 0; j < n - i; ++j) {
        ran[i + j] = ran_m[j];
    }

    for (i = 0; i < j; ++i) {
        each_state[j - 1 - i] = (uint32_t)(state_m[i]);
    }
    for (i = j; i < 8; ++i) {
        each_state[j + 7 - i] = (uint32_t)(r_state[i]);
    }
#else
    int i, j, k;
    uint32_t r_state[8];
    uint64_t s1[8], s2[8], s[8];
    double rnorm = 1.0 / static_cast<double>(MASK);

    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
    }

    for (i = 0; i < n; i+=8) {
        for (k = 0; k < 8; ++k) {
            s1[k] = 0;
            s2[k] = 0;
        }
        for (j = 0; j < 4; ++j) {
            for (k = 0; k < 8; ++k) {
                s1[k] += (uint64_t)(A8_OP_MATRIX[j * 8 + k]) * r_state[j];
                s2[k] += (uint64_t)(A8_OP_MATRIX[(j + 4) * 8 + k]) * r_state[j + 4];
            }
        }
        for (k = 0; k < 8 && i + k < n; ++k) { //only unroll not vectorized
            s[k] = (s1[k] & MASK) + (s1[k] >> 31) + (s2[k] & MASK) + (s2[k] >> 31);
            r_state[k] = (s[k] & MASK) + (s[k] >> 31);
            ran[i + k] = static_cast<double>(r_state[k] - 1) * rnorm;
        }
    }
    for (i = 0; i < k; ++i) {
        each_state[k - 1 - i] = r_state[i];
    }
    for (i = k; i < 8; ++i) {
        each_state[k + 7 - i] = r_state[i];
    }

#endif
}
    
inline void mrg8_vec::mrg8_vec_outer(double * ran, const int n)
{
    mrg8_vec_outer(ran, n, state);
}
    
inline double mrg8_vec::mrg8_vec_outer()
{
    double r;
    mrg8_vec_outer(&r, 1, state);
    return r;
}

inline double mrg8_vec::mrg8_vec_outer(uint32_t *new_state)
{
    double r;
    mrg8_vec_outer(&r, 1, new_state);
    return r;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
inline void mrg8_vec::mrg8_vec_inner_tp(double * ran, int n)
{
    int tnum = omp_get_max_threads();
    uint32_t next_state[8];
#pragma omp parallel
    {
        int each_n = ((n / tnum) / 8) * 8;
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

inline void mrg8_vec::mrg8_vec_outer_tp(double * ran, int n)
{
    int tnum = omp_get_max_threads();
    uint32_t next_state[8];
#pragma omp parallel
    {
        int each_n = ((n / tnum) / 8) * 8;
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

inline void mrg8_vec::mrg8_vec_outer_tp_small(double * ran, int each_n, int it)
{
    int tnum = omp_get_max_threads();
    uint32_t next_state[8];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int offset = each_n * tid;
        int start = offset * it;
        uint32_t *each_state = new uint32_t[8];
        jump_ahead(start, each_state);
        for (int i = 0; i < it; ++i) {
            mrg8_vec_outer(ran + offset, each_n, each_state);
        }
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

#endif
