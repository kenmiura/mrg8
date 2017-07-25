/*
 * stream.cpp
 * Evaluate Bandwidth of writing memory
 *  Created on: July 25, 2017
 *      Author: Yusuke
 */

#include <iostream>
#include <omp.h>

#define ITER 11

using namespace std;

void stream(double *a, int N)
{
    int i;
#pragma omp parallel for
    for (i = 0; i < N; ++i) {
        a[i] = i;
    }
}

int main(int argc, char **argv)
{
    int i;
    int N;
    double start, end, msec, ave_msec, bandwidth;
    double *a;

    cout << "Running on " << omp_get_max_threads() << "threads" << endl;

    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1024 * 1024;
    }

    cout << "Evaluating on " << N << " of 64-bit elements (= " << N * sizeof(double) << " [Bytes]" << endl;

    a = new double[N];

    ave_msec = 0;
    for (i = 0; i < ITER; ++i) {
        start = omp_get_wtime();
        stream(a, N);
        end = omp_get_wtime();
        msec = (end - start) * 1000;
        if (i > 0) {
            ave_msec += msec;
        }
    }

    ave_msec /= (ITER - 1);
    bandwidth = (double)N * sizeof(double) / 1024 / 1024 / ave_msec;
    cout << "StreamTest (Write only) : " << bandwidth << " [GB/sec], " << ave_msec << " [milli seconds]" << endl;

    delete[] a;
    
    return 0;
}
