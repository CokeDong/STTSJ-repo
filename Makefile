# Makefile example for compiling cuda and linking cuda to cpp:
SOURCELOC = 

UTILITYLOC =

NEWMOD =

PROGRAM = CUDASTTSJ

INCDIR= .
#
# Define the C compile flags
CCFLAGS = -g -m64 -I /usr/local/cuda/include -I ./header -O2 -std=c++11
CC = g++

# Define the Cuda compile flags
#
CUDAFLAGS= -O2 --gpu-architecture=compute_37 --gpu-code=compute_37 -m64 -use_fast_math -I ./header -std=c++11 -lineinfo --use-local-env -ccbin "g++" -cudart static --cl-version 2015
CUDACC= nvcc

# Define Cuda objects

#

CUDA = kernel.o

# Define the libraries

SYSLIBS= -lc
USRLIB  = -lcudart
# Define all object files

OBJECTS = \
	CUDASTTSJ_CPP.o\
	MBB.o\
	preprocess.o\
	STCell.o\
	STGrid.o\
	STInvertedList.o\
	STPoint.o\
	STTrajectory.o\
	STzorder.o\
	test.o\
	util.o

install: CUDASTTSJ


# Define Task Function Program

all: CUDASTTSJ

# Define what Modtools is


CUDASTTSJ: $(OBJECTS) $(CUDA)

	$(CUDACC) $(CUDAFLAGS) -o CUDASTTSJ -L/usr/local/cuda/lib64 -lcusparse -lcuda $(OBJECTS) $(CUDA) $(USRLIB) $(SYSLIBS)

CUDASTTSJ_CPP.o: main.cpp

	$(CC) $(CCFLAGS) -c main.cpp -o CUDASTTSJ_CPP.o

.cpp.o:

	$(CC) $(CCFLAGS) -c $<

CUDAINCDIR= /usr/local/cuda/include 

kernel.o: kernel.cu

	$(CUDACC) $(CUDAFLAGS) --compile -c kernel.cu -o kernel.o

clean:
	rm -rf *.o
#  end
