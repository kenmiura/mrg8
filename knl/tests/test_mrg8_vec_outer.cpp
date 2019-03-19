/*
 * MRG8-Vectorized Random Generator Outer product version
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
    int i, N;
    double *ran;
    double start, end, msec, ave_msec, mrng;
    uint32_t iseed;
    int tnum;

    tnum = omp_get_max_threads();

    cout << "Running on " << tnum << "threads" << endl;

    iseed = 13579;

    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1024 * 1024;
    }

    cout << "Generating " << N << " of 64-bit floating random numbers" << endl;

    mrg8_vec m(iseed);
    ran = (double *)_mm_malloc(sizeof(double) * N, 64);

    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.mrg8_vec_outer(ran, N);
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
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_VEC_OUTER: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, MRG8_VEC_OUTER, %d, %d, %f, %f\n", tnum, N, mrng, ave_msec);

    mrg8 mm(iseed);
    double *ans = (double *)_mm_malloc(sizeof(double) * N, 64);
    mm.rand(ans, N);
    bool flag = true;
    for (i = 0; i < N; ++i) {
        if (ran[i] != ans[i]) {
            flag = false;
            break;
        }
    }
    cout << i << ": " << flag << endl;

    mm.print_state();
    m.print_state();
    _mm_free(ans);

    _mm_free(ran);

    return 0;
}

