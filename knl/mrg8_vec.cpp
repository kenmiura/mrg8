/*
 * mrg8.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: June 29, 2017
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

#define AVX512

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
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8_vec::mrg8_vec_inner(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX512
    int i, j, k;
    double rnorm = 1.0 / static_cast<double>(MASK);
    uint64_t r_state[8];
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
            
            s_32m = _mm256_set_epi32((int)state2_m[7], (int)state2_m[6], (int)state2_m[5], (int)state2_m[4], (int)state2_m[3], (int)state2_m[2], (int)state2_m[1], (int)state2_m[0]);
            
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

            s_32m = _mm256_set_epi32((int)state1_m[7], (int)state1_m[6], (int)state1_m[5], (int)state1_m[4], (int)state1_m[3], (int)state1_m[2], (int)state1_m[1], (int)state1_m[0]);
            
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
            
        s_32m = _mm256_set_epi32((int)state2_m[7], (int)state2_m[6], (int)state2_m[5], (int)state2_m[4], (int)state2_m[3], (int)state2_m[2], (int)state2_m[1], (int)state2_m[0]);
            
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

        s_32m = _mm256_set_epi32((int)state1_m[7], (int)state1_m[6], (int)state1_m[5], (int)state1_m[4], (int)state1_m[3], (int)state1_m[2], (int)state1_m[1], (int)state1_m[0]);
            
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
// #elif defined AVX2
//     const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
    
//     int i, j, k;
//     double rnorm = 1.0 / static_cast<double>(MASK);
//     uint64_t r_state[8];
//     __m256i state11_m, state12_m, state21_m, state22_m, s1_m, s2_m, s_m, mask_m, a_m;
//     __m256d ran_m, rnorm_m;
//     __m128i s1_32m, s2_32m;
//     uint64_t s;
    
//     for (i = 0; i < 8; ++i) {
//         r_state[i] = (uint64_t)(each_state[7 - i]);
//     }
    
//     state11_m = _mm256_maskload_epi64(r_state, true_m);
//     state12_m = _mm256_maskload_epi64(r_state + 4, true_m);
//     mask_m = _mm256_set1_epi64x(MASK);
//     rnorm_m = _mm256_set1_pd(rnorm);
    
//     for (i = 0; i < n - 8; i+=8) {
//         if (((i >> 3) & 1) == 0) {
//             for (k = 0; k < 8; ++k) {
//                 a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8, true_m);
//                 s1_m = _mm256_mul_epu32(a_m, state11_m);
//                 a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8 + 4, true_m);
//                 s2_m = _mm256_mul_epu32(a_m, state12_m);

//                 s_m = _mm256_add_epi64(s1_m, s2_m);
//                 s1_m = _mm256_and_si256(s_m, mask_m);
//                 s2_m = _mm256_srli_epi64(s_m, 31);
//                 s_m = _mm256_add_epi64(s1_m, s2_m);
//                 s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

//                 if (k < 4) {
//                     state21_m[k] = s;
//                 }
//                 else {
//                     state21_m[k - 4] = s;
//                 }
//             }
//             s_m = _mm256_and_si256(state21_m, mask_m);
//             state21_m = _mm256_srli_epi64(state21_m, 31);
//             state21_m = _mm256_add_epi64(s_m, state21_m);
//             s_m = _mm256_and_si256(state22_m, mask_m);
//             state22_m = _mm256_srli_epi64(state22_m, 31);
//             state22_m = _mm256_add_epi64(s_m, state22_m);
            
//             s_m = _mm256_set_epi32((int)state22_m[3], (int)state22_m[2], (int)state22_m[1], (int)state22_m[0], (int)state21_m[3], (int)state21_m[2], (int)state21_m[1], (int)state21_m[0]);
            
//             s1_32m = mm256_extractf128_si256(s_m, 0);
//             s2_32m = mm256_extractf128_si256(s_m, 1);

//             ran_m = _mm256_cvtepi32_pd(s1_32m);
//             ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//             _mm256_store_pd(ran + i, ran_m);
//             ran_m = _mm256_cvtepi32_pd(s2_32m);
//             ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//             _mm256_store_pd(ran + i + 4, ran_m);
//         }
//         else {
//             for (k = 0; k < 8; ++k) {
//                 a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8, true_m);
//                 s1_m = _mm256_mul_epu32(a_m, state11_m);
//                 a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8 + 4, true_m);
//                 s2_m = _mm256_mul_epu32(a_m, state12_m);

//                 s_m = _mm256_add_epi64(s1_m, s2_m);
//                 s1_m = _mm256_and_si256(s_m, mask_m);
//                 s2_m = _mm256_srli_epi64(s_m, 31);
//                 s_m = _mm256_add_epi64(s1_m, s2_m);
//                 s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

//                 if (k < 4) {
//                     state11_m[k] = s;
//                 }
//                 else {
//                     state12_m[k - 4] = s;
//                 }
//             }
//             s_m = _mm256_and_si256(state11_m, mask_m);
//             state11_m = _mm256_srli_epi64(state11_m, 31);
//             state11_m = _mm256_add_epi64(s_m, state11_m);
//             s_m = _mm256_and_si256(state12_m, mask_m);
//             state12_m = _mm256_srli_epi64(state12_m, 31);
//             state12_m = _mm256_add_epi64(s_m, state12_m);

//             s_m = _mm256_set_epi32((int)state12_m[3], (int)state12_m[2], (int)state12_m[1], (int)state12_m[0], (int)state11_m[3], (int)state11_m[2], (int)state11_m[1], (int)state11_m[0]);

//             s1_32m = mm256_extractf128_si256(s_m, 0);
//             s2_32m = mm256_extractf128_si256(s_m, 1);

//             ran_m = _mm256_cvtepi32_pd(s1_32m);
//             ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//             _mm256_store_pd(ran + i, ran_m);
//             ran_m = _mm256_cvtepi32_pd(s2_32m);
//             ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//             _mm256_store_pd(ran + i + 4, ran_m);
//         }
//     }
    
//     if (((i >> 3) & 1) == 0) {
//         for (k = 0; k < (n - i); ++k) {
//             a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8, true_m);
//             s1_m = _mm256_mul_epu32(a_m, state11_m);
//             a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8 + 4, true_m);
//             s2_m = _mm256_mul_epu32(a_m, state12_m);

//             s_m = _mm256_add_epi64(s1_m, s2_m);
//             s1_m = _mm256_and_si256(s_m, mask_m);
//             s2_m = _mm256_srli_epi64(s_m, 31);
//             s_m = _mm256_add_epi64(s1_m, s2_m);
//             s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

//             if (k < 4) {
//                 state21_m[k] = s;
//             }
//             else {
//                 state21_m[k - 4] = s;
//             }
//         }
//         s_m = _mm256_and_si256(state21_m, mask_m);
//         state21_m = _mm256_srli_epi64(state21_m, 31);
//         state21_m = _mm256_add_epi64(s_m, state21_m);
//         s_m = _mm256_and_si256(state22_m, mask_m);
//         state22_m = _mm256_srli_epi64(state22_m, 31);
//         state22_m = _mm256_add_epi64(s_m, state22_m);
            
//         s_m = _mm256_set_epi32((int)state22_m[3], (int)state22_m[2], (int)state22_m[1], (int)state22_m[0], (int)state21_m[3], (int)state21_m[2], (int)state21_m[1], (int)state21_m[0]);
            
//         s1_32m = mm256_extractf128_si256(s_m, 0);
//         s2_32m = mm256_extractf128_si256(s_m, 1);

//         ran_m = _mm256_cvtepi32_pd(s1_32m);
//         ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//         _mm512_store_pd(ran + i, ran_m);
//         for (k = 0; k < n - i; ++k) {
//             if (k < 4) {
//                 ran[i + k] = ran_m[k];
//             }
//         }
        
//         ran_m = _mm256_cvtepi32_pd(s2_32m);
//         ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//         _mm512_store_pd(ran + i + 4, ran_m);
//         for (k = 4; k < n - i; ++k) {
//             ran[i + k] = ran_m[k];
//         }
            
//         for (j = k; j < 8; ++j) {
//             if (j < 4) {
//                 each_state[7 - (j - k)] = (uint32_t)(state11_m[j]);
//             }
//             else {
//                 each_state[7 - (j - k)] = (uint32_t)(state12_m[j]);
//             }
//         }
//         for (j = 0; j < k; ++j) {
//             if (j < 4) {
//                 each_state[k - j - 1] = (uint32_t)(state21_m[j]);
//             }
//             else {
//                 each_state[k - j - 1] = (uint32_t)(state22_m[j]);
//             }
//         }
//     }
//     else {
//         for (k = 0; k < (n - i); ++k) {
//             a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8, true_m);
//             s1_m = _mm256_mul_epu32(a_m, state11_m);
//             a_m = _mm256_maskload_epi64(A8_IP_MATRIX + k * 8 + 4, true_m);
//             s2_m = _mm256_mul_epu32(a_m, state12_m);

//             s_m = _mm256_add_epi64(s1_m, s2_m);
//             s1_m = _mm256_and_si256(s_m, mask_m);
//             s2_m = _mm256_srli_epi64(s_m, 31);
//             s_m = _mm256_add_epi64(s1_m, s2_m);
//             s = s_m[0] + s_m[1] + s_m[2] + s_m[3];

//             if (k < 4) {
//                 state11_m[k] = s;
//             }
//             else {
//                 state12_m[k - 4] = s;
//             }
//         }
//         s_m = _mm256_set_epi32((int)state12_m[3], (int)state12_m[2], (int)state12_m[1], (int)state12_m[0], (int)state11_m[3], (int)state11_m[2], (int)state11_m[1], (int)state11_m[0]);

//         s1_32m = mm256_extractf128_si256(s_m, 0);
//         s2_32m = mm256_extractf128_si256(s_m, 1);
            
//         ran_m = _mm256_cvtepi32_pd(s1_32m);
//         ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//         _mm256_store_pd(ran + i, ran_m);
//         for (k = 0; k < n - i; ++k) {
//             if (k < 4) {
//                 ran[i + k] = ran_m[k];
//             }
//         }
        
//         ran_m = _mm256_cvtepi32_pd(s2_32m);
//         ran_m = _mm256_mul_pd(ran_m, rnorm_m);
//         _mm256_store_pd(ran + i + 4, ran_m);
//         for (k = 4; k < n - i; ++k) {
//             ran[i + k] = ran_m[k];
//         }

//         for (j = k; j < 8; ++j) {
//             if (j < 4) {
//                 each_state[7 - (j - k)] = (uint32_t)(state21_m[j]);
//             }
//             else {
//                 each_state[7 - (j - k)] = (uint32_t)(state22_m[j]);
//             }
//         }
//         for (j = 0; j < k; ++j) {
//             if (j < 4) {
//                 each_state[k - j - 1] = (uint32_t)(state11_m[j]);
//             }
//             else {
//                 each_state[k - j - 1] = (uint32_t)(state12_m[j]);
//             }
//         }
//     }
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
            r_state[1 - target][k] = (s & MASK) + (s >> 31);
            ran[i + k] = static_cast<double>(r_state[1 - target][k]) * rnorm;
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
#ifdef AVX512
    int i, j;
    uint64_t r_state[8];
    __m256i state_32m;
    __m512i a_m, state_m, s_m, s1_m, s2_m, mask_m;
    __m512d ran_m, rnorm_m;
    double rnorm = 1.0 / static_cast<double>(MASK);

    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
    }

    mask_m = _mm512_set1_epi64(MASK);
    rnorm_m = _mm512_set1_pd(rnorm);

    for (i = 0; i < n - 8; i+=8) {
        s1_m = _mm512_set1_epi64(0);
        s2_m = _mm512_set1_epi64(0);

        for (j = 0; j < 4; ++j) {
            state_m = _mm512_set1_epi64((uint64_t)(r_state[j]));
            a_m = _mm512_load_epi64(A8_OP_MATRIX + j * 8);
            state_m = _mm512_mul_epu32(a_m, state_m);
            s1_m = _mm512_add_epi64(s1_m, state_m);

            state_m = _mm512_set1_epi64((uint64_t)(r_state[j + 4]));
            a_m = _mm512_load_epi64(A8_OP_MATRIX + (j + 4) * 8);
            state_m = _mm512_mul_epu32(a_m, state_m);
            s2_m = _mm512_add_epi64(s2_m, state_m);
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

        _mm512_store_epi64(r_state, state_m);

        state_32m = _mm256_set_epi32((int)state_m[7], (int)state_m[6], (int)state_m[5], (int)state_m[4], (int)state_m[3], (int)state_m[2], (int)state_m[1], (int)state_m[0]);

        ran_m = _mm512_cvtepi32_pd(state_32m);
        ran_m = _mm512_mul_pd(ran_m, rnorm_m);
        _mm512_store_pd(ran + i, ran_m);
        // _mm512_store_pd(ran, ran_m);
    }

    /* Fraction */
    s1_m = _mm512_set1_epi64(0);
    s2_m = _mm512_set1_epi64(0);

    for (j = 0; j < 4; ++j) {
        state_m = _mm512_set1_epi64((uint64_t)(r_state[j]));
        a_m = _mm512_load_epi64(A8_OP_MATRIX + j * 8);
        state_m = _mm512_mul_epu32(a_m, state_m);
        s1_m = _mm512_add_epi64(s1_m, state_m);

        state_m = _mm512_set1_epi64((uint64_t)(r_state[j + 4]));
        a_m = _mm512_load_epi64(A8_OP_MATRIX + (j + 4) * 8);
        state_m = _mm512_mul_epu32(a_m, state_m);
        s2_m = _mm512_add_epi64(s2_m, state_m);
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

    // _mm512_store_epi64(r_state, state_m);

    state_32m = _mm256_set_epi32((int)state_m[7], (int)state_m[6], (int)state_m[5], (int)state_m[4], (int)state_m[3], (int)state_m[2], (int)state_m[1], (int)state_m[0]);

    ran_m = _mm512_cvtepi32_pd(state_32m);
    ran_m = _mm512_mul_pd(ran_m, rnorm_m);
    for (j = 0; j < n - i; ++j) {
        ran[i + j] = ran_m[j];
        // ran[j] = ran_m[j];
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
            ran[i + k] = static_cast<double>(r_state[k]) * rnorm;
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
