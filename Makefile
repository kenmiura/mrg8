ICPC = icpc

FLAGS = -O3 -g -qopenmp
MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt

OBJ = mrg8.o test.o
OBJ_VEC = mrg8.o mrg8_vec.o test_vec.o

all:
	make mrg8
	make mrg8_vec

mrg8 : $(OBJ)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ)

mrg8_vec : $(OBJ_VEC)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ_VEC)

stream : stream.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ stream.o

%.o : %.cpp
	$(ICPC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o mrg8 mrg8_vec stream
