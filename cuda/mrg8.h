/*
 * mrg8.h
 *

MRG8 Random Number Generator Library (MRG8) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov <mailto:IPO@lbl.gov>.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.

 */

#include <vector>
#include <stdint.h> // for uint32_t and uint64_t
#include <iostream>
#include <cmath>

using namespace std;

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

	void seed_init(const uint32_t seed_val);
	void get_state(uint32_t st[8]) const;
	void set_state(const uint32_t st[8]);
	void print_state() const;
	void print_matrix(const uint32_t jm[8][8]) const;

    /* Sequential Random Generator for int32 ver.1 */
	void randint(uint32_t * iran, int n);
	uint32_t randint();

    /* Sequential Random Generator for double ver.1 */
	void rand(double * fran, int n);
	double rand();
    double operator() ()
    {
        return rand();
    }
    
    /* Sequential Random Generator for double ver.2 */
    void mrg8dnz2(double * ran, int n);

    /*  Thread parallel Random Generator */
	void jump_ahead(const short jump_val_bin[200], uint32_t *new_state);
	void jump_ahead(const uint64_t jump_val, uint32_t *new_state);
	/* void rand_tp(double *ran, int n); */
    /* void mrg8dnz2_tp(double *ran, int n); */

protected:
	const uint64_t MASK;//2^31 - 1
	const uint32_t COEFF0, COEFF1, COEFF2, COEFF3, COEFF4, COEFF5, COEFF6, COEFF7;
    
    bool isJumpMatrix;
    
	uint32_t iseed;
	/* std::vector<uint32_t> JUMP_MATRIX; */
    uint32_t *JUMP_MATRIX;

	uint32_t state[8];// state[0]  =  S[n-1], state[1] = S[n-2], ..., state[7] = S[n-8]
    
	void mcg64ni();
	void jump_calc(const short jump_val_bin[200], uint32_t jump_mat[8][8]);
	void read_jump_matrix();
	uint32_t bigDotProd(const uint32_t x[8], const uint32_t y[8]) const;
	void dec2bin(const uint64_t jval, short jump_val_bin[200]) const;

private:
	void randint(uint32_t * iran, int n, uint32_t *new_state);
	void rand(double * fran, int n, uint32_t *new_state);
    void mrg8dnz2(double * ran, int n, uint32_t *new_state);
};
