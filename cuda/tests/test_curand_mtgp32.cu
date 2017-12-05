/*
 * cuRAND-MTGP32 Random Generator - GPU
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

#include <curand.h>
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
    cout << "cuRAND_MTGP32: " << mrng << " [million rng/sec], " << ave_msec << " [milli seconds]" << endl;
    printf("EVALUATION, cuRAND_MTGP32, , , %d, %f, %f\n", N, mrng, ave_msec);

    cudaFree(d_ran);
    delete[] ran;
    for (i = 0; i < 2; ++i) {
        cudaEventDestroy(event[i]);
    }

    return 0;
}

