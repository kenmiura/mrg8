/*
 * mrg8.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 */

#include "mrg8.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8::mrg8(): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(
			1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(
			1927846343), iseed(0), JUMP_MATRIX(8 * 8 * 247) {
	mcg64ni();
	read_jump_matrix();
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mrg8::mrg8(const uint32_t seed_val): MAX_RND(2147483646), MASK(2147483647), COEFF0(1089656042), COEFF1(1906537547), COEFF2(1764115693), COEFF3(
		1304127872), COEFF4(189748160), COEFF5(1984088114), COEFF6(626062218), COEFF7(
		1927846343), iseed(seed_val), JUMP_MATRIX(8 * 8 * 247){
	mcg64ni();
	read_jump_matrix();
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
void mrg8::set_state(const uint32_t st[8]){
	 for (int i=0; i<8;++i)
    	 state[i] = st[i] ;
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
uint32_t mrg8::bigDotProd(const uint32_t x[8], const uint32_t y[8]) const {
	uint64_t s, s1, s2;
	s1 = 0;s2 = 0;s = 0;
	for (int q = 0; q < 4; ++q) {
		s1 += uint64_t(x[q]) * y[q];
		s2 += uint64_t(x[4 + q]) * y[4 + q];
	}
	s = (s1 & MASK) + (s1 >> 31) + (s2 & MASK) + ( s2 >> 31);
	s = (s & MASK) + (s >> 31);
	return ((s & MASK) + (s >> 31));
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reading pre-calculated jump matrix from file
// A^(2^j) mod (2^31 - 1) for j = 0, 1, ..., 246 where A is the 8x8 one-step jump matrix
// Each matrix is in column order, that is (A^(2^j))(r,c) = JUMP_MATRIX[j * 64 + r + 8*c]
void mrg8::read_jump_matrix(){
	std::ifstream infile;
	infile.open("jump_matrix.txt");
	if(infile.fail()){
		std::cerr << "jump_matrix.txt could not be opened! Terminating!"<<std::endl;
		exit(EXIT_FAILURE);
	}

	uint32_t t;
	for(int k=0;k<8*8*247;++k){
		infile >> t;
		JUMP_MATRIX[k] = t;
	}
    infile.close();
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Jumping the state ahead by a 200-bit value with the zero index as the LSB
void mrg8::jump_ahead(const short jump_val_bin[200]) {
	uint32_t jump_mat[8][8], new_state[8], rowVec[8];

	//calculating the jump matrix
	jump_calc(jump_val_bin, jump_mat);

	// Multiply the current state by jump_mat
	for (int r = 0; r < 8; ++r){
		for (int c = 0;c<8;++c){
			rowVec[c] = jump_mat[r][c];
		}
		new_state[r] = bigDotProd(rowVec, state);
	}

	for(int i=0;i<8;++i)
		state[i] = new_state[i];

}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Jumping the state ahead by a jum_val
void mrg8::jump_ahead(const uint64_t jump_val) {
     short jump_val_bin[200];
     dec2bin(jump_val, jump_val_bin);
     jump_ahead(jump_val_bin);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mrg8::dec2bin(const uint64_t jval, short jump_val_bin[200]) const {
	for (int nb = 0;nb < 200;++nb) {
		jump_val_bin[nb] = 0;
	}

	for (int nb = 0; nb < 64; ++nb){
		if (jval & (1ul << nb))
			jump_val_bin[nb] = 1;
	}

}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Calculating the jump matrix for a given 200-bit jump value
// jump_val_bin is the jump value  with jump_val_bin[0] as the LSB
void mrg8::jump_calc(const short jump_val_bin[200], uint32_t jump_mat[8][8]) {
	uint32_t tmp_mat[8][8], vec1[8], vec2[8];

	for (int r = 0; r < 8; ++r) {
		for (int c = 0; c < 8; ++c) {
			if (r == c)
				jump_mat[r][c] = 1;
			else
				jump_mat[r][c] = 0;
		}
	}

	for (int nb = 0;nb < 200;++nb) {
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
void mrg8::mcg64ni(){
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
void mrg8::randint(uint32_t * iran, int n){
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
		 z = bigDotProd(a, state);

		 state[7] = state[6];// S[n-8] = S[n-7]
		 state[6] = state[5];// S[n-7] = S[n-6]
		 state[5] = state[4];// S[n-6] = S[n-5]
		 state[4] = state[3];// S[n-5] = S[n-4]
		 state[3] = state[2];// S[n-4] = S[n-3]
		 state[2] = state[1];// S[n-3] = S[n-2]
		 state[1] = state[0];// S[n-2] = S[n-1]
		 state[0] = z;
		 iran[k] = state[0];// y[n] = S[n]
	}

}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
uint32_t mrg8::randint(){
	uint32_t r[1];
	randint(r, 1);
	return r[0];
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Generate uniform RVs in [0, 1)
void mrg8::rand(double * fran, int n){
	uint32_t * iran1 = new uint32_t[n];
	double rnorm = 1.0/static_cast<double>(MASK);
	randint(iran1, n);
	for(int i=0;i<n;++i){
	   fran[i] = static_cast<double>(iran1[i]) * rnorm;
	}
	delete [] iran1;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double mrg8::rand(){
	double r[1];
	rand(r, 1);
	return r[0];
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//void mrg8::mrg8dnz2(double * ran, int n){
//double rnorm = 4.6566128752458e-10l;
//int kmax = 1024, nn;
//uint64_t s, s1, s2;
//uint64_t mask = 2147483647ull;
//uint64_t a[8] = {1089656042ull, 1906537547ull, 1764115693ull, 1304127872ull, 189748160ull, 1984088114ull, 626062218ull, 1927846343ull};
//uint64_t x[1032]= {0};
////uint32_t xx[8];
//
//
//for (int k=0;k<8;++k){
//	x[kmax+k] = x[k];
//}
//
//nn = n % kmax;
//
//for(int j = 0; j < n - nn; j += kmax){
//	for (int k=kmax;k>=1;--k){
//		s1 = 0;
//		s2 = 0;
//		for(int i=0;i<4;++i){
//			s1 += a[i] * x[k+i];
//			s2 += a[i+4] * x[k+i+4];
//		}
//		s = (s1 & mask) + (s1 >> 31) + (s2 & mask) + (s2 >> 31);
//		s = (s & mask) + (s >> 31);
//		x[k-1] = (s & mask) + (s >> 31);
//		ran[j + kmax-k] = static_cast<double>(x[k-1]) * rnorm;
//	}
//
//	for(int k=0;k<8;++k){
//		x[kmax+k] = x[k];
//	}
//
//}
//
//if (nn > 0){
//
//	for (int i=0;i<nn;++i){
//		s1 = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + a[3] * x[3];
//		s2 = a[4] * x[4] + a[5] * x[5] + a[6] * x[6] + a[7] * x[7];
//		s = (s1 & mask) + (s1 >> 31) + (s2 & mask) + ( s2 >> 31);
//		s = (s & mask) + (s >> 31);
//		state[7] = state[6];
//		state[6] = state[5];
//		state[5] = state[4];
//		state[4] = state[3];
//		state[3] = state[2];
//		state[2] = state[1];
//		state[1] = state[0];
//		state[0] = (s & mask) + (s >> 31);
//		ran[n-nn+i] = static_cast<double>(state[0]) * rnorm;
//
//	}
//}
//
//}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
