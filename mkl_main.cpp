/*
 * main.cpp
 *
 *  Created on: June 27, 2017
 *      Author: Nagasaka
 */

#include "mrg8_vec.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <ctime>
#include <omp.h>
#include "mkl_vsl.h"
#include "errcheck.inc"

using namespace std;

#define ITER 5
#define DEBUG

void check_rand(const double *ran, const int N)
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

void mkl_rng(double *ran, const int n, const int method)
{
    const double a = 0.0, b = 1.0;
    uint32_t iseed = 13579;
    int tnum = omp_get_max_threads();

#pragma omp parallel
    {
        int errcode;
        int each_n = n / tnum;
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        VSLStreamStatePtr stream;
        if (tid == tnum - 1) {
            each_n = n - each_n * tid;
        }
        errcode = vslNewStream(&stream, VSL_BRNG_MRG32K3A, iseed);
        vslSkipAheadStream(stream, each_n * tid);
        CheckVslError(errcode);
        errcode = vdRngUniform(method, stream, each_n, ran + start, a, b);
        vslDeleteStream(&stream);
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int main(int argc, char **argv)
{
    int i, N;
    double *ran;
    double start, end, msec, ave_msec, mrng;
    int tnum = omp_get_max_threads();

    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1024 * 1024;
    }

    /* Thread-Parallel MKL RNG -accurate- */
    ave_msec = 0;
    ran = new double[N * ITER];
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran + i * N, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE);
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
    cout << "MKL_ACCURATE, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;
    delete[] ran;

    /* Thread-Parallel MKL RNG -fast- */
    ave_msec = 0;
    ran = new double[N * ITER];
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran + i * N, N, VSL_RNG_METHOD_UNIFORM_STD);
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
    cout << "MKL_FAST, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;
    delete[] ran;

}

