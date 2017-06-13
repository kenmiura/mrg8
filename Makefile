CC = icpc

FLAGS = -O3 -g
#-xMIC-AVX512 -qopt-report=5

OBJ = mrg8_jv.o

all:
	make mrg

mrg : $(OBJ)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $(OBJ)

%.o : %.cpp
	$(CC) -c $(FLAGS) $(INCLUDE) -o $@ $<

clean :
	rm mrg8_jv.o mrg
