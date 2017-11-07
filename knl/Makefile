ICPC = icpc

FLAGS = -O3 -g -qopenmp
INCLUDE = -I../common -I/opt/intel/include/icc

MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt

PREF_MRG = test_mrg
PREF_MKL = test_mkl

KNL_OBJ = mrg8.o mrg8_vec.o
HW_OBJ = mrg8.o mrg8_vec_hw.o

MRG_OBJ = $(HW_OBJ)
SRC = ./tests
MRG_TEST = $(wildcard $(SRC)/$(PREF_MRG)*.cpp)
MKL_TEST = $(wildcard $(SRC)/$(PREF_MKL)*.cpp)
MRG_TEST_TARGET = $(notdir $(MRG_TEST:.cpp=))
MKL_TEST_TARGET = $(notdir $(MKL_TEST:.cpp=))

BIN = ./bin

all:
	make test

test:
	make mrg_test
	make mkl_test

mrg_test: $(addprefix $(BIN)/,$(MRG_TEST_TARGET))
mkl_test: $(addprefix $(BIN)/,$(MKL_TEST_TARGET))

$(BIN)/$(PREF_MKL)%: $(SRC)/$(PREF_MKL)%.o
	mkdir -p $(BIN)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $^

$(BIN)/$(PREF_MRG)%: $(MRG_OBJ) $(SRC)/$(PREF_MRG)%.o
	mkdir -p $(BIN)
	$(ICPC) $(FLAGS) $(MRGLIBS) $(INCLUDE) -o $@ $^

stream : stream.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ stream.o

%.o : %.cpp
	$(ICPC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o $(BIN)/* $(OBJ_TEST)/*.o
