ICPC = icpc

FLAGS = -O3 -g -qopenmp
INCLUDE = -I../common -I/opt/intel/include/icc

MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt

MV_OBJ = mrg8.o mrg8_vec.o
SRC = ./tests
TEST = $(wildcard $(SRC)/*.cpp)
TEST_TARGET = $(notdir $(TEST:.cpp=))

BIN = ./bin

all:
	make test

test: $(addprefix $(BIN)/,$(TEST_TARGET))

$(BIN)/%: $(MV_OBJ) $(SRC)/%.o
	@echo $@
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $@ $^

%.o : %.cpp
	@echo $@
	$(ICPC) -c $(FLAGS) $(INCLUDE) -o $@ $<


# stream : stream.o
# 	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ stream.o



clean :
	rm -f *.o $(BIN)/* $(OBJ_TEST)/*.o
