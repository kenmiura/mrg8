#include <iostream>
#include <cuda.h>

#define ITER 11

using namespace std;

__global__ void stream(double *d_a, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) {
        return;
    }
    d_a[i] = i;
}

int main(int argc, char **argv)
{
    int i, N;
    double *a, *d_a;
    float msec, ave_msec, bandwidth;
    cudaEvent_t event[2];
    
    for (i = 0; i < 2; ++i) {
        cudaEventCreate(&(event[i]));
    }

    if (argc > 1) {
        N = atoi(argv[1]) * 1024 * 1024;
    }
    else {
        N = 1 * 1024 * 1024;
    }

    a = new double[N];
    cudaMalloc((void **)&d_a, sizeof(double) * N);

    ave_msec = 0;
    int GS, BS;
    BS = 512;
    GS = (N + BS - 1) / BS;
    for (i = 0; i < ITER; ++i) {
        cudaEventRecord(event[0], 0);

        stream<<<GS, BS>>>(d_a, N);

        cudaEventRecord(event[1], 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= (ITER - 1);
    bandwidth = (float)(N * sizeof(double)) / 1024 / 1024 / ave_msec ;
    cout << "StreamTest, " << N << ", " << bandwidth << ", " << ave_msec << endl;

    cudaFree(d_a);
    free(a);

    return 0;

}
