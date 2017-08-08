/*
 * mrg8.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 *  Updated on: Jul 15, 2017
 *      Author: Yusuke
 */

#include "mrg8.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

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
// initializing the state with a 32-bit seed
void mrg8::seed_init(const uint32_t seed_val)
{
	iseed = seed_val;
	mcg64ni();
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::get_state(uint32_t st[8]) const
{
	for (int k = 0; k < 8; ++k)
		st[k] = state[k];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::print_state() const
{
	std::cout << "(S[n-1], S[n-2], ..., S[n-8])= (";
    for (int i = 0; i < 7; ++i) {
        std::cout<<std::setw(10)<<state[i]<<", ";
    }
    std::cout << state[7]<<")"<<std::endl;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::print_matrix(const uint32_t jm[8][8]) const
{
	for (int i = 0; i < 8;++i) {
		for(int j = 0; j < 8; ++j) {
			std::cout<<std::setw(10)<<jm[i][j]<<" ";
        }
		std::cout << std::endl;
	}
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::set_state(const uint32_t st[8])
{
    for (int i = 0; i < 8; ++i)
        state[i] = st[i];
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
	s1 = 0;s2 = 0;s = 0;
	for (int q = 0; q < 4; ++q) {
		s1 += uint64_t(x[q]) * y[q];
		s2 += uint64_t(x[4 + q]) * y[4 + q];
	}
	s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + ( s2 >> 31);
	// s = (s & MASK) + (s >> 31);
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
	if(infile.fail()){
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

	// Calculating the jump matrix
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
	for (int nb = 0;nb < 200;++nb) {
		jump_val_bin[nb] = 0;
	}

	for (int nb = 0; nb < 64; ++nb) {
		if (jval & (1ul << nb)) {
			jump_val_bin[nb] = 1;
        }
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
			if (r == c) {
				jump_mat[r][c] = 1;
            }
			else {
				jump_mat[r][c] = 0;
            }
		}
	}

	for (int nb = 0; nb < 200; ++nb) {
		if (jump_val_bin[nb]) {
			for (int r = 0; r < 8; ++r){
				for (int c = 0; c < 8; ++c){
					for (int q = 0; q < 8; ++q) {
						vec1[q] = jump_mat[r][q];
						vec2[q] = JUMP_MATRIX[64*nb + q + 8*c];
					}
					tmp_mat[r][c] = bigDotProd(vec1, vec2);
				}
			}

			for (int r = 0;r < 8;++r)
				for (int c = 0;c < 8;++c)
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
	for(int k=0;k<8;++k){
		x = ia * x;
		tmp = (x >> 32);
		state[k] = (tmp>>1);
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

    for (int k=0;k<n;++k){
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
    randint(iran, n, state);
}

uint32_t mrg8::randint()
{
	uint32_t r[1];
	randint(r, 1, state);
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
    rand(ran, n, state);
}

double mrg8::rand()
{
	double r;
	rand(&r, 1, state);
	return r;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Generate uniform RVs in [0, 1)
void mrg8::mrg8dnz2(double * ran, int n, uint32_t *new_state)
{
    int i, j, k;
    double rnorm = 1.0 / static_cast<double>(MASK);
    int kmax = 1024;
    int nn;
    uint64_t s, s1, s2;
	uint32_t a[8], z;
    uint32_t x[kmax + 8];

	a[0] = COEFF0;
	a[1] = COEFF1;
	a[2] = COEFF2;
	a[3] = COEFF3;
	a[4] = COEFF4;
	a[5] = COEFF5;
	a[6] = COEFF6;
	a[7] = COEFF7;

    for (k = 0; k < 8; ++k) {
        x[kmax + k] = new_state[k];
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

    for (k = 0; k < 8; ++k) {
        new_state[k] = x[kmax + k];
    }
    if (nn > 0) {
        rand(ran + (n - nn), nn, new_state);
    }
}

void mrg8::mrg8dnz2(double * ran, int n)
{
    mrg8dnz2(ran, n, state);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Thread Parallel Random Number Generation with uniform RVs in [0, 1)
void mrg8::rand_tp(double * ran, int n)
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
        rand(ran + start, each_n, each_state);
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

void mrg8::mrg8dnz2_tp(double * ran, int n)
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
        mrg8dnz2(ran + start, each_n, each_state);
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

