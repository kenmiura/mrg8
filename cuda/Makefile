CC = nvcc

FLAGS = -O3 -g -arch=sm_37 -lcudart -lcurand
#-xMIC-AVX512 -qopt-report=5

OBJ = mrg8_cuda.o main.o

all:
	make mrg8

mrg8 : $(OBJ)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ)

stream : stream.o
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ stream.o

%.o : %.cu
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o mrg8 stream
