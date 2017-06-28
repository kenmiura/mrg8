CC = icpc

FLAGS = -O3 -g -fopenmp -axMIC-AVX512
MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt
#-xMIC-AVX512 -qopt-report=5

OBJ = mrg8_vec.o main.o

all:
	make mrg8

mrg8 : $(OBJ)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $(OBJ)

mkl_main : mkl_main.o
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ mkl_main.o

%.o : %.cpp
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm *.o mrg8 mkl_main
