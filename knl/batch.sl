#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -p regular
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,flat

#!/bin/sh

s_out="./result/0914_mrg_single_1.txt"
d_out="./result/0914_mrg_ddr.txt"
m_out="./result/0914_mrg_mcdram.txt"

for i in `seq 1 10`
do
    # Single generation
    ./bin/test_cpp_single >> $s_out
    ./bin/test_mrg8_single >> $s_out
    for th in 1 2 4 8 16 32 64 68 128 136 192 204 256 272
    do
        echo "Thread num: $th"
        OMP_NUM_THREADS=$th ./bin/test_mrg8_single_tp >> $s_out
        OMP_NUM_THREADS=$th ./bin/test_mrg8_single_tp $th >> $s_out
    done

    # Small RNGs generation
    for x in `seq 0 4`
    do
        size=$((1<<$x))
        echo $size
        numactl --membind=0 ./bin/test_mkl_mrg32k3a_a $size >> $d_out
        numactl --membind=0 ./bin/test_mkl_mt19937_a $size >> $d_out
        numactl --membind=0 ./bin/test_mkl_sfmt19937_a $size >> $d_out
        numactl --membind=0 ./bin/test_mkl_mt2203_a $size >> $d_out
        numactl --membind=0 ./bin/test_mkl_philox4x32x10_a $size >> $d_out
        numactl --membind=0 ./bin/test_mrg8_1 $size >> $d_out
        numactl --membind=0 ./bin/test_mrg8_2 $size >> $d_out
        numactl --membind=0 ./bin/test_mrg8_vec_inner $size >> $d_out
        numactl --membind=0 ./bin/test_mrg8_vec_outer $size >> $d_out

        numactl --membind=1 ./bin/test_mkl_mrg32k3a_a $size >> $m_out
        numactl --membind=1 ./bin/test_mkl_mt19937_a $size >> $m_out
        numactl --membind=1 ./bin/test_mkl_sfmt19937_a $size >> $m_out
        numactl --membind=1 ./bin/test_mkl_mt2203_a $size >> $m_out
        numactl --membind=1 ./bin/test_mkl_philox4x32x10_a $size >> $m_out
        numactl --membind=1 ./bin/test_mrg8_1 $size >> $m_out
        numactl --membind=1 ./bin/test_mrg8_2 $size >> $m_out
        numactl --membind=1 ./bin/test_mrg8_vec_inner $size >> $m_out
        numactl --membind=1 ./bin/test_mrg8_vec_outer $size >> $m_out
    done

    # Scalability (Strong)
    for th in 1 2 4 8 16 32 64 68 128 136 192 204 256 272
    do
        echo "Thread num: $th"
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt19937_a_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_sfmt19937_a_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mrg32k3a_a_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt2203_a_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_philox4x32x10_a_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_1_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_2_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_inner_tp >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_outer_tp >> $d_out

        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt19937_a_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_sfmt19937_a_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mrg32k3a_a_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt2203_a_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_philox4x32x10_a_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_1_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_2_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_inner_tp >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_outer_tp >> $m_out
    done
    
    # Scalability (Weak)
    for th in 1 2 4 8 16 32 64 68 128 136 192 204 256 272
    do
        echo "Thread num: $th"
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt19937_a_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_sfmt19937_a_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mrg32k3a_a_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt2203_a_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_philox4x32x10_a_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_1_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_2_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_inner_tp $th >> $d_out
        OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_outer_tp $th >> $d_out

        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt19937_a_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_sfmt19937_a_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mrg32k3a_a_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt2203_a_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_philox4x32x10_a_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_1_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_2_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_inner_tp $th >> $m_out
        OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_outer_tp $th >> $m_out
    done
    
    # Large RNGs generation
    for x in `seq 5 9`
    do
        size=$((1<<$x))
        echo $size
        for th in 68 132 204 272
        do
            echo "Thread num: $th"
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt19937_a_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_sfmt19937_a_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mrg32k3a_a_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_mt2203_a_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mkl_philox4x32x10_a_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_1_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_2_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_inner_tp $size >> $d_out
            OMP_NUM_THREADS=$th numactl --membind=0 ./bin/test_mrg8_vec_outer_tp $size >> $d_out

            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt19937_a_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_sfmt19937_a_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mrg32k3a_a_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_mt2203_a_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mkl_philox4x32x10_a_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_1_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_2_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_inner_tp $size >> $m_out
            OMP_NUM_THREADS=$th numactl --membind=1 ./bin/test_mrg8_vec_outer_tp $size >> $m_out
        done
    done
done
