CC = nvcc

FLAGS = -O3 -g -arch=sm_37 -lcudart -lcurand
INCLUDE = -I../common

OBJ = mrg8.o mrg8_cuda.o
BIN = ./bin

all :
	make mrg8

mrg8 : $(OBJ)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $(OBJ)

test :
	make test_curand_mrg32k3a
	make test_curand_mt19937
	make test_curand_philox4_32_10
	make test_curand_xorwow
	make test_curand_mtgp32
	make test_mrg8_inner
	make test_mrg8_outer

test_curand_mrg32k3a : test_curand_mrg32k3a.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ test_curand_mrg32k3a.o

test_curand_mt19937 : test_curand_mt19937.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ test_curand_mt19937.o

test_curand_philox4_32_10 : test_curand_philox4_32_10.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ test_curand_philox4_32_10.o

test_curand_xorwow : test_curand_xorwow.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ test_curand_xorwow.o

test_curand_mtgp32 : test_curand_mtgp32.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ test_curand_mtgp32.o

test_mrg8_inner : $(OBJ) test_mrg8_inner.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) test_mrg8_inner.o

test_mrg8_outer : $(OBJ) test_mrg8_outer.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) test_mrg8_outer.o

stream : stream.o
	$(CC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ stream.o

%.o : %.cu
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

%.o : %.cpp
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o mrg8 stream $(BIN)/*
