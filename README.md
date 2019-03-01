MRG8
======

## Versions
2.1 (March, 2019)

## Introduction
This is a pseudorandom number generator based on the 8th order primitive polynomial modulo 2^31-1. This polynomial was provided by Prof. P. L’Ecuyer of Montreal University.
The MRG8 firstly appeared in (1), and further optimized version was proposed in (2) later.

(1) Kenta Hongo, Ryo Maezono, Kenichi Miura, "Random number generators tested on quantum Monte Carlo simulations", Journal of computational chemistry, 2010

(2) Yusuke Nagasaka, Akira Nukada, Satoshi Matsuoka, Kenichi Miura, John Shalf, "MRG8 - Random Number Generation for the Exascale Era", The Platform for Advanced Scientific Computing Conference (PASC18), July 2018 (to be appeared)

## Components and Requirement
### C/Fortran version (./original)
 - Nothing special
### CUDA version (./cuda)
 - CUDA: >= 5.0
 - Compute capability: >= 3.5
 - Use of cuRAND library for test program (MRG8 itself does not require)
### KNL version (./knl)
 - OpenMP
 - Use of Intel MKL library for test program (MRG8 itself does not require)

## Run sample program
### Preparation
To use this library, the first thing you need to do is to modify the Makefile with correct CUDA/Intel Compiler installation path. The compute capability for CUDA is also appropriately set.
### CUDA version
 - Compile with Makefile: make test
   - This command generates all test executable file including cuRAND library.
 - Run on KNL processor: './bin/test_mrg8_outer 16'
   - First argument is the number of RNGs to be generated (* 2^20). Above example means "MRG8 optimized for NVIDIA GPU generates 2^24 of RNGs".

### KNL version
 - Compile with Makefile: make test
   - This command generates all test executable file including Intel MKL
 - Run on KNL processor: './bin/test_mrg8_vec_outer_tp 16'
   - First argument is the number of RNGs to be generated (* 2^20). Above example means "MRG8 with AVX-512 and full threads generates 2^24 of RNGs".

## To use with your own program
### CUDA version
'make build' provides static library of MRG8-CUDA, generating 'libmrg8.a'. Copy 'mrg8.h' and 'mrg8_cuda.h' to your codes. To use the library, add include mrg8_vec.h in your code, and compile with 'libmrg8.a'.

### KNL version
Copy 'mrg8.h' and 'mrg8_vec.h' to your codes. To use the library, include file (mrg8_vec.h) in your code.

## Notes
More detailed document is in ./original/ReadmeMRG8.doc
