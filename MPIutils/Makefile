#GNU makefile
# USE gmake IF IT DOESNT WORK CORRECTLY ON YOUR SYSTEM!
# This makefile uses gnumake specific conditional syntax
# just comment out the conditional stuff if you can't find gmake

#----------------------------------------------------------------------
# Figure out what kind of system we are on without resorting to config.sub
APRUN := $(shell which aprun)
ifeq ($(APRUN), /usr/bin/aprun)
UNAME = CrayXT
else
UNAME := $(shell uname | perl -pe 's/(sn\d\d\d\d|jsimpson)/UNICOS\/mk/')
endif
#----------------------------------------------------------------------
# now apply correct compilers and build flags for systems we know about
ifeq ($(UNAME), Darwin)
CXX = g++
FLAGS = -O2
MPILIB = -lmpi -lmpi_cxx
endif

ifeq ($(UNAME), AIX)
CXX=mpCC_r
FLAGS = -qhot 
# optional if you want to use GPFS-specific MPI-IO hints
# FLAGS = -qhot -DHAS_GPFS
endif

ifeq ($(UNAME), CrayXT)
CXX = CC 
LUSTREHOME = /opt
# optional if you want to have benchmark automatically manipulate striping
# rather than using the defaults of the directory
# FLAGS = -O2 -DHAS_LUSTRE -I$(LUSTREHOME)/include
# MPILIB = -L$(LUSTREHOME)/lib -llustre_user
FLAGS = -O2
endif

ifeq ($(UNAME), Linux)
CXX = mpicxx
FLAGS = -O2
# FLAGS = -O2 -DHAS_GPFS
endif


all: test

PatternIO: MPIutils.o Timer.o Timeval.o PatternIO.o
	$(CXX) $(FLAGS) -o PatternIO PatternIO.o Timer.o Timeval.o MPIutils.o $(MPILIB) -lm

MPIutils.o: MPIutils.cc MPIutils.hh
	$(CXX) -c $(FLAGS) MPIutils.cc

Timer.o: Timer.cc Timer.hh Timeval.hh
	$(CXX) -c $(FLAGS) Timer.cc

Timeval.o: Timeval.cc Timeval.hh
	$(CXX) -c $(FLAGS) Timeval.cc

PatternIO.o: PatternIO.cxx Timeval.hh Timer.hh MPIutils.hh
	$(CXX) -c $(FLAGS) PatternIO.cxx

test: testibarrier testbarrier testisend testiisend testisend.send testissend testiissend
	echo $(APRUN)
	echo $(UNAME)

Parameters.o: Parameters.hh Parameters.cc
	$(CXX) $(CFLAGS) -c Parameters.cc

testibarrier: testibarrier.o Timer.o MPIutils.o vatoi.o Parameters.o
	$(CXX) $(CFLAGS) -o testibarrier testibarrier.o MPIutils.o Parameters.o Timer.o vatoi.o -lmpi 

testibarrier.o: testibarrier.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testibarrier.cc

testbarrier: testbarrier.o Timer.o MPIutils.o vatoi.o Parameters.o 
	$(CXX) $(CFLAGS) -o testbarrier testbarrier.o  Parameters.o MPIutils.o Timer.o vatoi.o -lmpi 

testbarrier.o: testbarrier.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testbarrier.cc

testiissend: testiissend.o Timer.o MPIutils.o vatoi.o Parameters.o 
	$(CXX) $(CFLAGS) -o testiissend testiissend.o MPIutils.o  Parameters.o Timer.o vatoi.o -lmpi 

testiissend.o: testiissend.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testiissend.cc

testissend: testissend.o Timer.o MPIutils.o vatoi.o  Parameters.o 
	$(CXX) $(CFLAGS) -o testissend testissend.o  Parameters.o MPIutils.o Timer.o vatoi.o -lmpi 

testissend.o: testissend.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testissend.cc

testiisend: testiisend.o Timer.o MPIutils.o vatoi.o Parameters.o 
	$(CXX) $(CFLAGS) -o testiisend testiisend.o MPIutils.o  Parameters.o Timer.o vatoi.o -lmpi 

testiisend.o: testiisend.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testiisend.cc

testisend.send: testisend.send.o Timer.o MPIutils.o vatoi.o Parameters.o 
	$(CXX) $(CFLAGS) -o testisend.send testisend.send.o MPIutils.o  Parameters.o Timer.o vatoi.o -lmpi 

testisend.send.o: testisend.send.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testisend.send.cc

testisend: testisend.o Timer.o MPIutils.o vatoi.o Parameters.o 
	$(CXX) $(CFLAGS) -o testisend testisend.o MPIutils.o  Parameters.o Timer.o vatoi.o -lmpi 

testisend.o: testisend.cc Parameters.hh
	$(CXX) $(CFLAGS) -c testisend.cc

clean: 
	rm -f *.o
