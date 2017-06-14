/*
 * mrg8.h
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 */

#ifndef MRG8_H_
#define MRG8_H_
#include <vector>
#include <stdint.h> // for uint32_t and uint64_t

// S[n] = (A0* S[n-1] + A1 * S[n-2] + ...+ A7 * S[n-8]) mod (2^31-1)
// y[n] = S[n]
// Coefficients Ak are carefully chosen 31-bit integers
// S[k]'s are 31-bit integers
// Period of this random sequence is (2 ^ 31 - 1)^ 8 - 1 =~ 4.5 * 10^74


class mrg8 {
public:
	const uint32_t MAX_RND;

	mrg8();
	mrg8(const uint32_t seed_val);

    void randint(uint32_t *iran, int n, uint32_t *new_state);
	void randint(uint32_t *iran, int n);
	uint32_t randint();
	void rand(double *ran, int n, uint32_t *new_state);
	void rand(double *ran, int n);
	double rand();
    
    void mrg8dnz2(double *ran, int n, uint32_t *new_state);
    void mrg8dnz2(double *ran, int n);

    void mrg8dnz_inner(double *ran, int n, uint32_t *each_state);
    void mrg8dnz_inner(double *ran, int n);

    void mrg8dnz_outer(double * ran, int n, uint32_t *each_state);
    void mrg8dnz_outer(double * ran, int n);

	void rand_tp(double *ran, int n);
    void mrg8dnz2_tp(double *ran, int n);
    void mrg8dnz_inner_tp(double *ran, int n);
    void mrg8dnz_outer_tp(double * ran, int n);

	void seed_init(const uint32_t seed_val);
	void get_state(uint32_t st[8]) const;
	void set_state(const uint32_t st[8]);
	void jump_ahead(const short jump_val_bin[200]);
	void jump_ahead(const uint64_t jump_val);
    void jump_ahead(const short jump_val_bin[200], uint32_t *new_state);
    void jump_ahead(const uint64_t jump_val, uint32_t *new_state);

	void print_state() const;
	void print_matrix(const uint32_t jm[8][8]) const;

private:
	const uint64_t MASK;//2^31 - 1
	const uint32_t COEFF0, COEFF1, COEFF2, COEFF3, COEFF4, COEFF5, COEFF6, COEFF7;
    
	uint32_t iseed;
	std::vector<uint32_t> JUMP_MATRIX;
	uint32_t state[8];// state[0]  =  S[n-1], state[1] = S[n-2], ..., state[7] = S[n-8]
    
    bool isJumpMatrix;

	void mcg64ni();
	void jump_calc(const short jump_val_bin[200], uint32_t jump_mat[8][8]);
	void read_jump_matrix();
	uint32_t bigDotProd(const uint32_t x[8], const uint32_t y[8]) const;
	void dec2bin(const uint64_t jval, short jump_val_bin[200]) const;
};

#endif /* MRG8_H_ */
