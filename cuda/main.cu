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

#include "mrg8_cuda.h"
#include <curand.h>

using namespace std;

#define ITER 10
// #define DEBUG

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
    double *ran, *d_ran;
    float msec, ave_msec, mrng;
    uint32_t iseed;
    cudaEvent_t event[2];
    
    for (i = 0; i < 2; ++i) {
        cudaEventCreate(&(event[i]));
    }

    iseed = 13579;
    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1024 * 1024;
    }

    ran = new double[N];
    // cout << "Allocate array on device" << endl;
    cudaMalloc((void **)&d_ran, sizeof(double) * N);

    // cout << "Start Random Generate on GPU" << endl;
    // cout << "Algorithm, Size, million RNG/sec, average milli second" << endl << endl;

#if 1
    /* cuRAND-MRG32K3A Random Generator - GPU */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        cudaEventRecord(event[0], 0);

        curandGenerator_t prngGPU;
        curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MRG32K3A);
        curandSetPseudoRandomGeneratorSeed(prngGPU, iseed);
        curandGenerateUniformDouble(prngGPU, d_ran, N);
        
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
        if (i == 0) {
            cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
            check_rand(ran, N);
        }
#endif
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << ", cuRAND_MRG32K3A, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* cuRAND-MT19937 Random Generator - GPU */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        cudaEventRecord(event[0], 0);

        curandGenerator_t prngGPU;
        curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MT19937);
        curandSetPseudoRandomGeneratorSeed(prngGPU, iseed);
        curandGenerateUniformDouble(prngGPU, d_ran, N);
        
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
        if (i == 0) {
            cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
            check_rand(ran, N);
        }
#endif
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << ", cuRAND_MT19937, " << N << ", " << mrng << ", " << ave_msec << endl;

    /* cuRAND-MTGP32 Random Generator - GPU */
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        cudaEventRecord(event[0], 0);

        curandGenerator_t prngGPU;
        curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
        curandSetPseudoRandomGeneratorSeed(prngGPU, iseed);
        curandGenerateUniformDouble(prngGPU, d_ran, N);
        
        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
        if (i == 0) {
            cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
            check_rand(ran, N);
        }
#endif
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    mrng = (double)(N) / ave_msec / 1000;
    cout << ", cuRAND_MTGP32, " << N << ", " << mrng << ", " << ave_msec << endl;
#endif

    mrg8 m(iseed);

#if 0
    /* Sequential Random Generator - ver1*/
    struct timeval start, end;
    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        gettimeofday(&start, NULL);
        m.rand(ran, N);
        gettimeofday(&end, NULL);
        msec = (float)(end.tv_sec - start.tv_sec) * 1000 + (float)(end.tv_usec - start.tv_usec) / 1000;
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
    cout << "MRG8_SEQ1, " << N << ", " << mrng << ", " << ave_msec << endl;
#endif

    for (int TNUM = 64; TNUM <= 1024 * 1024; TNUM *= 2) {
        /* MRG8-CUDA inner */
        ave_msec = 0;
        for (i = 0; i < ITER; ++i) {
            cudaEventRecord(event[0], 0);
            m.mrg8_inner(d_ran, N, TNUM);
            cudaEventRecord(event[1], 0);
            cudaThreadSynchronize();
            cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
            if (i == 0) {
                cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
                check_rand(ran, N);
            }
#endif
            if (i > 0) {
                ave_msec += msec;
            }
        }
        ave_msec /= (ITER - 1);
        mrng = (double)(N) / ave_msec / 1000;
        cout << TNUM << ", ";
        cout << "MRG8_CUDA_inner, " << N << ", " << mrng << ", " << ave_msec << endl;

        /* MRG8-CUDA outer_32 */
        for (int BS = 32; BS < 1024; BS = BS << 1) {
            if (BS > TNUM) continue;
            ave_msec = 0;
            for (i = 0; i < ITER; ++i) {
                cudaEventRecord(event[0], 0);
                m.mrg8_outer_32(d_ran, N, BS, TNUM);
                cudaEventRecord(event[1], 0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
                if (i == 0) {
                    cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
                    check_rand(ran, N);
                }
#endif
                if (i > 0) {
                    ave_msec += msec;
                }
            }
            ave_msec /= (ITER - 1);
            mrng = (double)(N) / ave_msec / 1000;
            cout << TNUM << ", ";
            cout << "MRG8_CUDA_outer_32_" << BS << ", " << N << ", " << mrng << ", " << ave_msec << endl;
        }
    
        /* MRG8-CUDA outer_64 */
        ave_msec = 0;
        for (i = 0; i < ITER; ++i) {
            cudaEventRecord(event[0], 0);
            m.mrg8_outer_64(d_ran, N, TNUM);
            cudaEventRecord(event[1], 0);
            cudaThreadSynchronize();
            cudaEventElapsedTime(&msec, event[0], event[1]);
#ifdef DEBUG
            if (i == 0) {
                cudaMemcpy(ran, d_ran, sizeof(double) * N, cudaMemcpyDeviceToHost);
                check_rand(ran, N);
            }
#endif
            if (i > 0) {
                ave_msec += msec;
            }
        }
        ave_msec /= (ITER - 1);
        mrng = (double)(N) / ave_msec / 1000;
        cout << TNUM << ", ";
        cout << "MRG8_CUDA_outer_64, " << N << ", " << mrng << ", " << ave_msec << endl;
    }
    
    cudaFree(d_ran);
    delete[] ran;
    for (i = 0; i < 2; ++i) {
        cudaEventDestroy(event[i]);
    }

    return 0;
}

