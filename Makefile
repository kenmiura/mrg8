CC = icpc

FLAGS = -O3 -g -fopenmp
#-xMIC-AVX512 -qopt-report=5

OBJ = mrg8_vec.o main.o

all:
	make mrg8

mrg8 : $(OBJ)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $(OBJ)

%.o : %.cpp
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm *.o mrg8
