#!/bin/bash -l
 
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p debug
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,flat
 
#!/bin/sh

d_out="./result/0709_mrg_nj_ddr.csv"
m_out="./result/0709_mrg_nj_mcdram.csv"

for i in `seq 1 5`
do
    # for size in 68
    # do
    #     ./mrg8_seq $size >> $d_out
    #     numactl --membind=1 ./mrg8_seq $size >> $m_out
    # done

    for size in 1 2 3 4
    do
        for tnum in 1 2 3 4
        do
            OMP_NUM_THREADS=$tnum ./mrg8_tp $size >> $d_out
            OMP_NUM_THREADS=$tnum numactl --membind=1 ./mrg8_tp $size >> $m_out
        done
    done
done
