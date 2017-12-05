/*
 * mrg8.h
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 * Updated on: June 29, 2017
 *      Author: Yusuke
 */

#include <vector>
#include <stdint.h> // for uint32_t and uint64_t
#include "mrg8.h"

// S[n] = (A0* S[n-1] + A1 * S[n-2] + ...+ A7 * S[n-8]) mod (2^31-1)
// y[n] = S[n]
// Coefficients Ak are carefully chosen 31-bit integers
// S[k]'s are 31-bit integers
// Period of this random sequence is (2 ^ 31 - 1)^ 8 - 1 =~ 4.5 * 10^74

class mrg8_cuda : public mrg8
{
public:

	mrg8_cuda();
	mrg8_cuda(const uint32_t seed_val);

    void seed_init(const uint32_t seed_val);

    void mrg8_inner(double *d_ran, const int n, const int TNUM);
    void mrg8_outer_32(double *d_ran, const int n);
    void mrg8_outer_32(double *d_ran, const int n, const int TNUM);
    void mrg8_outer_32(double *d_ran, const int n, const int BS, const int TNUM);

    void state_cpy_DtH();

private:
    uint32_t *JUMP_MATRIX_R, *JUMP_MATRIX_8s_32, *JUMP_MATRIX_8s_64;
    uint32_t *d_JUMP_MATRIX, *d_JUMP_MATRIX_8s_32, *d_JUMP_MATRIX_8s_64;
    /* uint32_t *d_JUMP_MATRIX, *d_JUMP_MATRIX_8s_64; */
    uint32_t *d_state;
    
	void mcg64ni();
    void reverse_jump_matrix();
    void set_jump_matrix_8s_32();
    void set_jump_matrix_8s_64();
};
