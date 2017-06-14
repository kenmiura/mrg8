/*
 * mrg8.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: June 14, 2017
 *      Author: Nagasaka
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
mrg8::mrg8(): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(1927846343), iseed(0), JUMP_MATRIX(8 * 8 * 247), isJumpMatrix(false)
{
	mcg64ni();
	read_jump_matrix();
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8::mrg8(const uint32_t seed_val): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(1927846343), iseed(seed_val), JUMP_MATRIX(8 * 8 * 247), isJumpMatrix(false)
{
	mcg64ni();
	read_jump_matrix();
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
    // s = ((s & MASK) + (s >> 31));
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
    uint64_t long_state[8];

	if(iseed == 0)
		iseed = 97531;

	x = iseed;
	for (int k = 0; k < 8; ++k) {
		x = ia * x;
		tmp = (x >> 32);
		state[k] = (tmp>>1);
        long_state[k] = (tmp>>1);
	}
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
void mrg8::mrg8dnz2(double * ran, int n, uint32_t *new_state)
{
    int i, j, k;
    double rnorm = 1.0 / static_cast<double>(MASK);
    int kmax = 1024;
    int nn;
    uint64_t s, s1, s2;
	uint32_t a[8], z;
    uint32_t x[1032]= {0};

	a[0] = COEFF0;
	a[1] = COEFF1;
	a[2] = COEFF2;
	a[3] = COEFF3;
	a[4] = COEFF4;
	a[5] = COEFF5;
	a[6] = COEFF6;
	a[7] = COEFF7;

    for (k = 0; k < 8; ++k) {
        x[k] = new_state[k];
        x[kmax + k] = x[k];
    }
    
    nn = n % kmax;
    
    for (j = 0; j < n - nn; j += kmax) {
        for (k = kmax - 1; k >= 0; --k) {
            s1 = 0;
            s2 = 0;
            for (i = 0; i < 4; ++i) {
                s1 += uint64_t(a[i]) * x[k + 1 + i];
                s2 += uint64_t(a[i + 4]) * x[k + 5 + i];
            }
            s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
            x[k] = (s & MASK) + (s >> 31);
            ran[j + kmax - 1 - k] = static_cast<double>(x[k]) * rnorm;
        }
        for (k = 0; k < 8; ++k) {
            x[kmax + k] = x[k];
        }
    }

    if (nn > 0) {
        for (i = 0; i < nn; ++i) {
            s1 = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + a[3] * x[3];
            s2 = a[4] * x[4] + a[5] * x[5] + a[6] * x[6] + a[7] * x[7];
            s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
            x[7] = x[6];
            x[6] = x[5];
            x[5] = x[4];
            x[4] = x[3];
            x[3] = x[2];
            x[2] = x[1];
            x[1] = x[0];
            x[0] = (s & MASK) + (s >> 31);
            ran[n - nn + i] = static_cast<double>(x[0]) * rnorm;
        }
    }
}

void mrg8::mrg8dnz2(double * ran, int n)
{
    uint32_t *new_state = new uint32_t[8];
    for (int i = 0; i < 8; ++i) {
        new_state[i] = state[i];
    }
    mrg8dnz2(ran, n, new_state);
    delete[] new_state;

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::mrg8dnz_inner(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX512
    int i, j, k;
    uint64_t a8[64];
    double rnorm = 1.0 / static_cast<double>(MASK);
    uint64_t r_state[8];
    __m512i state1_m, state2_m, s1_m, s2_m, s_m, mask_m, a_m;
    __m512d ran_m, rnorm_m;
    __m256i s_32m;
    uint64_t s;
    
    read_jump_matrix();
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            a8[(7 - i) * 8 + (7 - j)] = (uint64_t)(JUMP_MATRIX[8 * 8 * 3 + i + j * 8]);
        }
    }
    for (i = 0; i < 8; ++i) {
        r_state[i] = (uint64_t)(each_state[7 - i]);
    }

    state1_m = _mm512_load_epi64(r_state);
    mask_m = _mm512_set1_epi64(MASK);
    rnorm_m = _mm512_set1_pd(rnorm);

    for (i = 0; i < n; i+=8) {
        if (((i >> 3) & 1) == 0) {
            for (k = 0; k < 8; ++k) {
                a_m = _mm512_load_epi64(a8 + k * 8);
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
                a_m = _mm512_load_epi64(a8 + k * 8);
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
#else
    int i, j, k;
    uint32_t a8[64];
    uint32_t r_state[2][8];
    uint64_t s1, s2, s;
    double rnorm = 1.0 / static_cast<double>(MASK);
    int target;
    read_jump_matrix();
    for (i = 0; i < 8; ++i) {
        for (j = 0; j < 8; ++j) {
            a8[(7 - i) * 8 + (7 - j)] = JUMP_MATRIX[8 * 8 * 3 + i + j * 8];
        }
    }
    for (i = 0; i < 8; ++i) {
        r_state[0][i] = state[7 - i];
    }
    
    for (i = 0; i < n; i+=8) {
        target = (i >> 3) & 1;
        for (k = 0; k < 8; ++k) {
            s1 = 0;
            s2 = 0;
            for (j = 0; j < 4; ++j) {
                s1 += (uint64_t)(a8[k * 8 + j]) * r_state[target][j];
                s2 += (uint64_t)(a8[k * 8 + j + 4]) * r_state[target][j + 4];
            }
            s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + (s2 >> 31);
            r_state[1 - target][k] = (s & MASK) + (s >> 31);
            ran[i + k] = static_cast<double>(r_state[1 - target][k]) * rnorm;
        }
    }
#endif
}

void mrg8::mrg8dnz_inner(double * ran, int n)
{
    uint32_t *new_state = new uint32_t[8];
    for (int i = 0; i < 8; ++i) {
        new_state[i] = state[i];
    }
    mrg8dnz_inner(ran, n, new_state);
    delete[] new_state;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::mrg8dnz_outer(double * ran, int n, uint32_t *each_state)
{
#ifdef AVX512
    int i, j;
    uint64_t a8[64];
    uint64_t r_state[8];
    __m256i state_32m;
    __m512i a_m, state_m, s_m, s1_m, s2_m, mask_m;
    __m512d ran_m, rnorm_m;
    double rnorm = 1.0 / static_cast<double>(MASK);

    read_jump_matrix();
    for (i = 0; i < 64; ++i) {
        a8[i] = (uint64_t)(JUMP_MATRIX[8 * 8 * 4 - 1 - i]);
    }
    for (i = 0; i < 8; ++i) {
        r_state[i] = each_state[7 - i];
    }

    mask_m = _mm512_set1_epi64(MASK);
    rnorm_m = _mm512_set1_pd(rnorm);

    for (i = 0; i < n; i+=8) {
        s1_m = _mm512_set1_epi64(0);
        s2_m = _mm512_set1_epi64(0);

        for (j = 0; j < 4; ++j) {
            state_m = _mm512_set1_epi64((uint64_t)(r_state[j]));
            a_m = _mm512_load_epi64(a8 + j * 8);
            state_m = _mm512_mul_epu32(a_m, state_m);
            s1_m = _mm512_add_epi64(s1_m, state_m);

            state_m = _mm512_set1_epi64((uint64_t)(r_state[j + 4]));
            a_m = _mm512_load_epi64(a8 + (j + 4) * 8);
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

    }
#else
    uint32_t a8[64];
    uint32_t r_state[8];
    uint64_t s1[8], s2[8], s[8];
    double rnorm = 1.0 / static_cast<double>(MASK);

    read_jump_matrix();
    for (int i = 0; i < 64; ++i) {
        a8[i] = JUMP_MATRIX[8 * 8 * 4 - 1 - i];
    }
    for (int i = 0; i < 8; ++i) {
        r_state[i] = state[7 - i];
    }

    for (int i = 0; i < n; i+=8) {
        for (int k = 0; k < 8; ++k) {
            s1[k] = 0;
            s2[k] = 0;
        }
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 8; ++k) {
                s1[k] += (uint64_t)(a8[j * 8 + k]) * r_state[j];
                s2[k] += (uint64_t)(a8[(j + 4) * 8 + k]) * r_state[j + 4];
            }
        }
        for (int k = 0; k < 8; ++k) { //only unroll not vectorized
            s[k] = (s1[k] & MASK) + (s1[k] >> 31) + (s2[k] & MASK) + (s2[k] >> 31);
            r_state[k] = (s[k] & MASK) + (s[k] >> 31);
            ran[i + k] = static_cast<double>(r_state[k]) * rnorm;
        }
    }
#endif
}
    
void mrg8::mrg8dnz_outer(double * ran, int n)
{
    uint32_t *new_state = new uint32_t[8];
    for (int i = 0; i < 8; ++i) {
        new_state[i] = state[i];
    }
    mrg8dnz_outer(ran, n, new_state);
    delete[] new_state;
}
    
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::rand_tp(double * ran, int n)
{
    read_jump_matrix();

    int tnum = omp_get_max_threads();
    int each_n = n / tnum;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        jump_ahead(start, each_state);
        rand(ran + start, each_n, each_state);
        delete[] each_state;
    }
}

void mrg8::mrg8dnz2_tp(double * ran, int n)
{
    read_jump_matrix();

    int tnum = omp_get_max_threads();
    int each_n = n / tnum;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        jump_ahead(start, each_state);
        mrg8dnz2(ran + start, each_n, each_state);
        delete[] each_state;
    }
}

void mrg8::mrg8dnz_inner_tp(double * ran, int n)
{
    read_jump_matrix();

    int tnum = omp_get_max_threads();
    int each_n = n / tnum;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        jump_ahead(start, each_state);
        mrg8dnz_inner(ran + start, each_n, each_state);
        delete[] each_state;
    }
}

void mrg8::mrg8dnz_outer_tp(double * ran, int n)
{
    read_jump_matrix();

    int tnum = omp_get_max_threads();
    int each_n = n / tnum;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        uint32_t *each_state = new uint32_t[8];
        jump_ahead(start, each_state);
        mrg8dnz_outer(ran + start, each_n, each_state);
        delete[] each_state;
    }
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
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
