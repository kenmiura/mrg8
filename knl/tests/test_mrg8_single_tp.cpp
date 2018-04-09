/*
 * MRG8 Sequential Random Generator - (non-array) - Thread parallel
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

    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);

        start = omp_get_wtime();

        uint32_t next_state[8];
#pragma omp parallel
        {
            int each_n = N / tnum;
            int tid = omp_get_thread_num();
            int start = each_n * tid;
            uint32_t *each_state = new uint32_t[8];
            if (tid == (tnum - 1)) {
                each_n = N - each_n * tid;
            }
            m.jump_ahead(start, each_state);
            for (int j = 0; j < each_n; ++j) {
                ran[start + j] = m.rand(each_state);
            }
            if (tid == tnum - 1) {
                for (int j = 0; j < 8; ++j) {
                    next_state[j] = each_state[j];
                }
            }
            delete[] each_state;
        }
        m.set_state(next_state);
        
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
    cout << "MRG8_SINGLE_TP: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, MRG8_SINGLE_TP, %d, %d, %f, %f\n", tnum, N, mrng, ave_msec);
    
    _mm_free(ran);

    return 0;
}

