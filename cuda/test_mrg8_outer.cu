/*
 * MRG8 Outer product version
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
#include <stdint.h>

#include "mrg8_cuda.h"
#include <rng_test.h>

using namespace std;

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

    cout << "Generating " << N << " of 64-bit floating random numbers" << endl;

    ran = new double[N];
    cudaMalloc((void **)&d_ran, sizeof(double) * N);

    mrg8_cuda m(iseed);
    
    for (int TNUM = 64; TNUM <= 1024 * 1024; TNUM *= 2) {
        /* MRG8-CUDA outer_32 */
        for (int BS = 32; BS < 1024; BS = BS << 1) {
            if (BS > TNUM) continue;
            ave_msec = 0;
            for (i = 0; i < ITER; ++i) {
                m.seed_init(iseed);
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

            cout << "MRG8_CUDA_outer with " << TNUM << " threads (thread block size = " << BS << "): " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
        }
    }
    
    cudaFree(d_ran);
    delete[] ran;
    for (i = 0; i < 2; ++i) {
        cudaEventDestroy(event[i]);
    }

    return 0;
}

