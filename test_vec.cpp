/*
 * main.cpp
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

#include "mrg8_vec.h"

using namespace std;

#define ITER 10
#define DEBUG

void check_rand(double *ran, int N)
{
    int i;
    double ave, var;

    ave = 0.0;
    for (i = 0; i < N; i++) {
        ave += ran[i];
    }
    ave /= (double)N;
    var = 0.0;
    for (i = 0; i < N; i++) {
        var += (ran[i] - ave) * (ran[i] - ave);
    }
    var /= (double)N;
    
    cout << "Arithmetic mean: " << ave << endl;
    cout << "Standard deviation: " << sqrt(var) << endl;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    ran = new double[N];

    /* Vectorized Random Generator - inner */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.mrg8_vec_inner(ran, N);
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
    cout << "MRG8_VEC_INNER: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;

    /* Vectorized Random Generator - outer */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        double r = 0;
        start = omp_get_wtime();
        for (int j = 0; j < N; ++j) {
            r = m();
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
    cout << "MRG8_VEC_OUTER (non-array): " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;

    /* Vectorized Random Generator - outer */
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

    /* Thread-Parallel and Vectorized Random Generator - inner */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.mrg8_vec_inner_tp(ran, N);
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
    cout << "MRG8_VEC_INNER_TP: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;

    /* Thread-Parallel and Vectorized Random Generator - outer */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        m.seed_init(iseed);
        start = omp_get_wtime();
        m.mrg8_vec_outer_tp(ran, N);
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
    cout << "MRG8_VEC_OUTER_TP: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    
    delete[] ran;

    return 0;
}

