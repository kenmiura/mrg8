ICPC = icpc

FLAGS = -O3 -g -qopenmp
INCLUDE = -I../common

MKLROOT = /opt/intel/mkl
MKLLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lmkl_rt

OBJ = mrg8.o
OBJ_VEC = mrg8.o mrg8_vec.o
OBJ_TEST = ./tests
BIN = ./bin

all:
	make mrg8
	make mrg8_vec

test:
	make test_mrg8_single
	make test_mrg8_1
	make test_mrg8_2
	make test_mkl_mrg32k3a_a
	make test_mkl_mrg32k3a_f
	make test_mkl_mt19937_a
	make test_mkl_mt19937_f
	make test_mkl_sfmt19937_a
	make test_mkl_sfmt19937_f
	make test_mrg8_1_tp
	make test_mrg8_2_tp
	make test_mkl_mrg32k3a_a_tp
	make test_mkl_mrg32k3a_f_tp
	make test_mkl_mt19937_a_tp
	make test_mkl_mt19937_f_tp
	make test_mkl_sfmt19937_a_tp
	make test_mkl_sfmt19937_f_tp
	make test_mrg8_vec_inner
	make test_mrg8_vec_outer
	make test_mrg8_vec_outer_single
	make test_mrg8_vec_inner_tp
	make test_mrg8_vec_outer_tp

test_mrg8_single : $(OBJ) $(OBJ_TEST)/test_mrg8_single.o
	mkdir -p $(BIN)
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) $(OBJ_TEST)/test_mrg8_single.o

test_mrg8_1 : $(OBJ) $(OBJ_TEST)/test_mrg8_1.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) $(OBJ_TEST)/test_mrg8_1.o

test_mrg8_2 : $(OBJ) $(OBJ_TEST)/test_mrg8_2.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) $(OBJ_TEST)/test_mrg8_2.o

test_mkl_mrg32k3a_a : $(OBJ_TEST)/test_mkl_mrg32k3a_a.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mrg32k3a_a.o

test_mkl_mrg32k3a_f : $(OBJ_TEST)/test_mkl_mrg32k3a_f.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mrg32k3a_f.o

test_mkl_mt19937_a : $(OBJ_TEST)/test_mkl_mt19937_a.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mt19937_a.o

test_mkl_mt19937_f : $(OBJ_TEST)/test_mkl_mt19937_f.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mt19937_f.o

test_mkl_sfmt19937_a : $(OBJ_TEST)/test_mkl_sfmt19937_a.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_sfmt19937_a.o

test_mkl_sfmt19937_f : $(OBJ_TEST)/test_mkl_sfmt19937_f.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_sfmt19937_f.o

test_mrg8_1_tp : $(OBJ) $(OBJ_TEST)/test_mrg8_1_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) $(OBJ_TEST)/test_mrg8_1_tp.o

test_mrg8_2_tp : $(OBJ) $(OBJ_TEST)/test_mrg8_2_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ) $(OBJ_TEST)/test_mrg8_2_tp.o

test_mkl_mrg32k3a_a_tp : $(OBJ_TEST)/test_mkl_mrg32k3a_a_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mrg32k3a_a_tp.o

test_mkl_mrg32k3a_f_tp : $(OBJ_TEST)/test_mkl_mrg32k3a_f_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mrg32k3a_f_tp.o

test_mkl_mt19937_a_tp : $(OBJ_TEST)/test_mkl_mt19937_a_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mt19937_a_tp.o

test_mkl_mt19937_f_tp : $(OBJ_TEST)/test_mkl_mt19937_f_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_mt19937_f_tp.o

test_mkl_sfmt19937_a_tp : $(OBJ_TEST)/test_mkl_sfmt19937_a_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_sfmt19937_a_tp.o

test_mkl_sfmt19937_f_tp : $(OBJ_TEST)/test_mkl_sfmt19937_f_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_TEST)/test_mkl_sfmt19937_f_tp.o

test_mrg8_vec_inner : $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_inner.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_inner.o

test_mrg8_vec_outer : $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer.o

test_mrg8_vec_outer_single : $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer_single.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer_single.o

test_mrg8_vec_inner_tp : $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_inner_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_inner_tp.o

test_mrg8_vec_outer_tp : $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer_tp.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ $(OBJ_VEC) $(OBJ_TEST)/test_mrg8_vec_outer_tp.o

stream : stream.o
	$(ICPC) $(FLAGS) $(MKLLIBS) $(INCLUDE) -o $(BIN)/$@ stream.o

%.o : %.cpp
	$(ICPC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -f *.o $(BIN)/* $(OBJ_TEST)/*.o
