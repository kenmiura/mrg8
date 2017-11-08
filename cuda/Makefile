CC = nvcc

FLAGS = -O3 -g -arch=sm_60 -lcudart -lcurand
INCLUDE = -I../common -I.

PREF_MRG = test_mrg
PREF_CURAND = test_curand

MRG_OBJ = mrg8.o mrg8_cuda.o
SRC = ./tests
MRG_TEST = $(wildcard $(SRC)/$(PREF_MRG)*.cu)
CURAND_TEST = $(wildcard $(SRC)/$(PREF_CURAND)*.cu)
MRG_TEST_TARGET = $(notdir $(MRG_TEST:.cu=))
CURAND_TEST_TARGET = $(notdir $(CURAND_TEST:.cu=))

BIN = ./bin

all :
	make test

test :
	make mrg_test
	make curand_test

mrg_test: $(addprefix $(BIN)/,$(MRG_TEST_TARGET))
curand_test: $(addprefix $(BIN)/,$(CURAND_TEST_TARGET))

$(BIN)/$(PREF_MRG)%: $(MRG_OBJ) $(SRC)/$(PREF_MRG)%.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^

$(BIN)/$(PREF_CURAND)%: $(SRC)/$(PREF_CURAND)%.o
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^

%.o : %.cu
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

%.o : %.cpp
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

stream : stream.o
	$(CC) $(FLAGS)  $(INCLUDE) -o $@ stream.o

clean :
	rm -f *.o mrg8 stream $(BIN)/*
