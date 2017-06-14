/*
 * main.cpp
 *
 *  Created on: June 14, 2017
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
    int tnum = omp_get_max_threads();

    iseed = 13579;
    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1024 * 1024;
    }

    /* Sequential Random Generator - ver1*/
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.rand(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
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
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz2(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
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
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz_inner(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
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
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz_outer(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_VEC_OUTER, " << "1, " << N << ", " << mrng << " * 10^6" << endl;

    /* Thread-Parallel Sequential Random Generator - ver1 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.rand_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ1_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6" << endl;

    /* Thread-Parallel Sequential Random Generator - ver2 */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz2_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_SEQ2_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6" << endl;

    /* Thread-Parallel and Vectorized Random Generator - inner */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz_inner_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_VEC_INNER_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6" << endl;

    /* Thread-Parallel and Vectorized Random Generator - outer */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        mrg8 m(iseed);
        ran = new double[N];
        start = omp_get_wtime();
        m.mrg8dnz_outer_tp(ran, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
#ifdef DEBUG
        if (i == 0) {
            check_rand(ran, N);
        }
#endif
        delete[] ran;
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << "MRG8_VEC_OUTER_TP, " << tnum << ", " << N << ", " << mrng << " * 10^6" << endl;
}

