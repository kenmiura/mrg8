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

    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1024 * 1024;
    }

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
    cout << "StreamTest, " << N << ", " << bandwidth << ", " << ave_msec << endl;

    delete[] a;
    
    return 0;
}
