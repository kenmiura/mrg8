/*
 * Sequential Random Generator - ver1 (non-array)
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

#include <random>

#include <rng_test.h>

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

    ran = new double[N];

    mt19937 mt(0);
    uniform_real_distribution<double> mt_uni(0, 1);
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        for (int j = 0; j < N; ++j) {
            ran[j] = mt_uni(mt);
        }
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
    cout << "C++11: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, C++11, %d, %d, %f, %f\n", tnum, N, mrng, ave_msec);
    
    delete[] ran;

    return 0;
}

