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
#include "mkl_vsl.h"
#include "errcheck.inc"

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

void mkl_rng(double *ran, const int n, const int method)
{
    const double a = 0.0, b = 1.0;
    uint32_t iseed = 13579;

    int errcode;
    VSLStreamStatePtr stream;
    errcode = vslNewStream(&stream, VSL_BRNG_MRG32K3A, iseed);
    CheckVslError(errcode);
    errcode = vdRngUniform(method, stream, n, ran, a, b);
    vslDeleteStream(&stream);
}

void mkl_rng_tp(double *ran, const int n, const int method)
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
    uint32_t iseed;
    int tnum = omp_get_max_threads();

    iseed = 13579;
    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1000 * 1000;
    }

    mrg8 m(iseed);
    ran = new double[N];

    /* Sequential Random Generator - ver1*/
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.rand(ran, N);
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
    cout << "MRG8_SEQ1, " << "1, " << N << ", " << mrng << " * 10^6" << endl;

    /* Sequential Random Generator - ver2*/
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.mrg8dnz2(ran, N);
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
    cout << "MRG8_SEQ2, " << "1, " << N << ", " << mrng << " * 10^6" << endl;

    /* Vectorized Random Generator - inner */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
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
    cout << "MRG8_VEC_INNER, " << "1, " << N << ", " << mrng << " * 10^6" << endl;

    /* Vectorized Random Generator - outer */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
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
    cout << "MRG8_VEC_OUTER, " << "1, " << N << ", " << mrng << " * 10^6" << endl;

    /* MKL RNG -accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE);
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

    /* MKL RNG -fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD);
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
    
    /* Thread-Parallel Sequential Random Generator - ver1 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
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
    cout << "MRG8_SEQ1_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;

    /* Thread-Parallel Sequential Random Generator - ver2 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.mrg8dnz2_tp(ran, N);
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
    cout << "MRG8_SEQ2_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;

    /* Thread-Parallel and Vectorized Random Generator - inner */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
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
    cout << "MRG8_VEC_INNER_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;

    /* Thread-Parallel and Vectorized Random Generator - outer */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
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
    cout << "MRG8_VEC_OUTER_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;

    /* Thread-Parallel MKL RNG -accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE);
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
    cout << "MKL_ACCURATE_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;

    /* Thread-Parallel MKL RNG -fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD);
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
    cout << "MKL_FAST_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6, " << msec << endl;
    
    delete[] ran;

    return 0;
}

