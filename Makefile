ICPC = icpc

FLAGS = -O3 -g -qopenmp
MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt

OBJ = mrg8.o test.o

all:
	make mrg8

mrg8 : $(OBJ)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ)

mrg8_tp : $(OBJ)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ)

stream : stream.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ stream.o

%.o : %.cpp
	$(ICPC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o mrg8 
