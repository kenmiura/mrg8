/*
 * MRG8 Sequential Random Generator ver.1 Thread Parallel
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
#include "../mrg8.h"

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

    mrg8 m(iseed);
    ran = (double *)_mm_malloc(sizeof(double) * N, 64);

    /* Sequential Random Generator - ver1*/
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.rand_tp(ran, N);
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
    cout << "MRG8_1_TP: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, MRG8_1_TP, %d, %d, %f, %f\n", tnum, N, mrng, ave_msec);

    _mm_free(ran);

    return 0;
}

