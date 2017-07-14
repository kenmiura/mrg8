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

#include "mrg8.h"
#include "mkl_vsl.h"
#include "errcheck.inc"

using namespace std;

#define ITER 10
#define DEBUG

void check_rand(double *ran, int N)
{
#ifdef DEBUG
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
#endif
}

void mkl_rng(double *ran, const int n, const int method, const int brng)
{
    const double a = 0.0, b = 1.0;
    uint32_t iseed = 13579;

    int errcode;
    VSLStreamStatePtr stream;
    errcode = vslNewStream(&stream, brng, iseed);
    CheckVslError(errcode);
    errcode = vdRngUniform(method, stream, n, ran, a, b);
    vslDeleteStream(&stream);
}

void mkl_rng_tp(double *ran, const int n, const int method, const int brng)
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
        errcode = vslNewStream(&stream, brng, iseed);
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
    int tnum;

    tnum = omp_get_max_threads();

    iseed = 13579;
    if (argc > 1) {
        N = atoi(argv[1]) * 1000 * 1000;
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
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ1, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Sequential Random Generator - ver2*/
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.mrg8dnz2(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ2, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* MKL RNG -VSL_BRNG_MRG32K3A_Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_MRG32K3A);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MRG32K3A_ACCURATE, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* MKL RNG -VSL_BRNG_MRG32K3A_Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_MRG32K3A);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MRG32K3A_FAST, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;
    
    /* MKL RNG -VSL_BRNG_SFMT19937_Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_SFMT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_SFMT19937_ACCURATE, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* MKL RNG -VSL_BRNG_SFMT19937_Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_SFMT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_SFMT19937_FAST, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;
    
    /* MKL RNG -VSL_BRNG_MT19937_Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_MT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MT19937_ACCURATE, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* MKL RNG -VSL_BRNG_MT19937_Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_MT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MT19937_FAST, " << "1, " << N << ", " << mrng << ", " << ave_msec << endl;
    
    /* Thread-Parallel Sequential Random Generator - ver1 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.rand_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ1_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel Sequential Random Generator - ver2 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        m.mrg8dnz2_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ2_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -MRG32K3A-Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_MRG32K3A);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MRG32K3A_ACCURATE_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -VSL_BRNG_MRG32K3A-Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_MRG32K3A);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MRG32K3A_FAST_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -VSL_BRNG_SFMT19937-Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_SFMT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_SFMT19937_ACCURATE_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -VSL_BRNG_SFMT19937-Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_SFMT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_SFMT19937_FAST_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -VSL_BRNG_MT19937-Accurate- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, VSL_BRNG_MT19937);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i == 0) {
            check_rand(ran, N);
        }
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MKL_VSL_BRNG_MT19937_ACCURATE_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;

    /* Thread-Parallel MKL RNG -VSL_BRNG_MT19937-Fast- */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        mkl_rng_tp(ran, N, VSL_RNG_METHOD_UNIFORM_STD, VSL_BRNG_MT19937);
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
    cout << "MKL_VSL_BRNG_MT19937_FAST_TP, " << tnum << ", " << N << ", " << mrng << ", " << ave_msec << endl;
    
    delete[] ran;

    return 0;
}

