#include <iostream>
#include <stdint.h>
#include <mkl_vsl.h>

void mkl_rng(double *ran, const int n, const int method, const int brng)
{
    const double a = 0.0, b = 1.0;
    uint32_t iseed = 13579;

    int errcode;
    VSLStreamStatePtr stream;
    errcode = vslNewStream(&stream, brng, iseed);
    if (errcode != VSL_ERROR_OK) {
        cout << "Error Creating NewStream" << endl;
        exit(1);
    }
    errcode = vdRngUniform(method, stream, n, ran, a, b);
    if (errcode != VSL_ERROR_OK) {
        cout << "Error RNG Uniform" << endl;
        exit(1);
    }
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
        if (errcode != VSL_ERROR_OK) {
            cout << "Error Creating NewStream" << endl;
            exit(1);
        }
        vslSkipAheadStream(stream, start);
        errcode = vdRngUniform(method, stream, each_n, ran + start, a, b);
        if (errcode != VSL_ERROR_OK) {
            cout << "Error RNG Uniform" << endl;
            exit(1);
        }
        vslDeleteStream(&stream);
    }
}

void mkl_rng_tp_small(double *ran, const int each_n, int it, const int method, const int brng)
{
    const double a = 0.0, b = 1.0;
    uint32_t iseed = 13579;
    int tnum = omp_get_max_threads();
#pragma omp parallel
    {
        int errcode;
        int tid = omp_get_thread_num();
        int offset = each_n * tid;
        int start = offset * it;
        VSLStreamStatePtr stream;
        errcode = vslNewStream(&stream, brng, iseed);
        if (errcode != VSL_ERROR_OK) {
            cout << "Error Creating NewStream" << endl;
            exit(1);
        }
        vslSkipAheadStream(stream, start);
        for (int i = 0; i < it; ++i) {
            errcode = vdRngUniform(method, stream, each_n, ran + offset, a, b);
        }
        if (errcode != VSL_ERROR_OK) {
            cout << "Error RNG Uniform" << endl;
            exit(1);
        }
        vslDeleteStream(&stream);
    }
}

void mkl_rng_tp_sub(double *ran, const int n, const int sub_n, const int method, const int brng)
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
        int itr = each_n / sub_n;
        if (tid == tnum - 1) {
            each_n = n - each_n * tid;
        }
        errcode = vslNewStream(&stream, brng, iseed);
        if (errcode != VSL_ERROR_OK) {
            cout << "Error Creating NewStream" << endl;
            exit(1);
        }
        vslSkipAheadStream(stream, each_n * tid);
        for (int i = 0; i < itr; ++i) {
            errcode = vdRngUniform(method, stream, sub_n, ran + start, a, b);
        }

        if (errcode != VSL_ERROR_OK) {
            cout << "Error RNG Uniform" << endl;
            exit(1);
        }
        vslDeleteStream(&stream);
    }
}

