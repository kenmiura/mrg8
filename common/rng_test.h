
/***********************
MRG8 Random Number Generator Library (MRG8) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov <mailto:IPO@lbl.gov>.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.
***********************8/

#include <iostream>
#include <cmath>

#define ITER 10
#define DEBUG

using namespace std;

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

