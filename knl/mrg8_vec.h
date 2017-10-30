/*
 * mrg8.h
 *
 *  Created on: Apr 6, 2015
 *      Author: aghasemi
 * Updated on: June 29, 2017
 *      Author: Yusuke
 */

#include <vector>
#include <stdint.h>
#include "mrg8.h"

class mrg8_vec : public mrg8
{
public:

	mrg8_vec();
	mrg8_vec(const uint32_t seed_val);

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

private:
    uint64_t A8_IP_MATRIX[64];
    uint64_t A8_OP_MATRIX[64];
    void mrg8_vec_inner(double *ran, int n, uint32_t *each_state);
    void mrg8_vec_outer(double * ran, int n, uint32_t *each_state);
};

