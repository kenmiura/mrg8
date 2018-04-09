/*
 * MRG8-Vectorized Random Generator Outer product version - Thread Parallel
 *
 *  Created on: June 29, 2017
 *      Author: Yusuke
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <ctime>
#include <omp.h>
#include <stdint.h>

#include <rng_test.h>
#include "../mrg8_vec.h"

using namespace std;

int main(int argc, char **argv)
{
    int i, N, it;
    double *ran;
    double start, end, msec, ave_msec, mrng;
    uint32_t iseed;
    int tnum;

    tnum = omp_get_max_threads();

    cout << "Running on " << tnum << "threads" << endl;

    iseed = 13579;

    if (argc > 2) {
        N = atoi(argv[1]);
        it = atoi(argv[2]);
    }
    else {
        N = 512;
        it = 1000;
    }
    
    cout << "Generating " << N << " of 64-bit floating random numbers " << it << " times" << endl;
    
    mrg8_vec m(iseed);
    ran = (double *)_mm_malloc(sizeof(double) * N * tnum, 64);

    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.mrg8_vec_outer_tp_small(ran, N, it);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N * it * tnum) / ave_msec / 1000;
    cout << "MRG8_VEC_OUTER_TP: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, MRG8_VEC_OUTER_TP, %d, %d, %f, %f\n", tnum, N, mrng, ave_msec);

    _mm_free(ran);

    return 0;
}

