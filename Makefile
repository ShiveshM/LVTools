# Compiler
CC=gcc
CXX=g++
AR=ar
LD=g++

DYN_SUFFIX=.dylib
DYN_OPT=-dynamiclib -install_name $(LIBnuSQUIDS)/$(DYN_PRODUCT) -compatibility_version $(VERSION) -current_version $(VERSION)

VERSION=1.0.0
PREFIX=/data/icecube/software/LVTools_paackage/LVTools/


#PATH_nuSQUIDS=$(shell pwd)
PATH_nuSQUIDS=${NUSQUIDS}
PATH_SQUIDS=${SQUIDS}

INCSQUIDS=$(SQUIDS)/include
LIBSQUIDS=$(SQUIDS)/lib
INCnuSQUIDS=$(NUSQUIDS)/include
LIBnuSQUIDS=$(NUSQUIDS)/lib

MAINS_SRC=$(wildcard mains/*.cpp)
MAINS=$(patsubst mains/%.cpp,bin/%.exe,$(MAINS_SRC))
#$(EXAMPLES_SRC:.cpp=.exe)

CXXFLAGS= -g -std=c++11 -I./inc

# Directories

GSL_CFLAGS=-I${GSL_2_2}/include
GSL_LDFLAGS=-L${GSL_2_2}/lib -lgsl -lgslcblas -lm
HDF5_CFLAGS=-I/usr/include
HDF5_LDFLAGS=-L${ANACONDA}/lib -lhdf5_hl -lhdf5 -lz -ldl -lm
SQUIDS_CFLAGS=-I${INCSQUIDS} -I${GSL_2_2}/include
SQUIDS_LDFLAGS=-L${LIBSQUIDS} -L${GSL_2_2}/lib -lSQuIDS -lgsl -lgslcblas -lm
PHYSTOOLS_LDFLAGS=-L${PHYSTOOLS}/lib -lPhysTools
BOOST_LFLAGS=-L${BOOST}/lib
BOOST_CFLAGS=-I${BOOST}/include

# FLAGS
CFLAGS= -std=c99 -O3 -fPIC -I$(INCnuSQUIDS) -I${PHYSTOOLS}/include ${BOOST_CFLAGS} $(SQUIDS_CFLAGS) $(GSL_CFLAGS) $(HDF5_CFLAGS)
LDFLAGS= -Wl,-rpath -Wl,$(LIBnuSQUIDS) -L$(LIBnuSQUIDS)
LDFLAGS+= $(SQUIDS_LDFLAGS) $(GSL_LDFLAGS) $(PHYSTOOLS_LDFLAGS) $(BOOST_LFLAGS) -lboost_system $(HDF5_LDFLAGS) 

# Compilation rules
all: $(MAINS)

bin/%.exe : mains/%.cpp mains/%.o mains/lbfgsb.o mains/linpack.o
	$(CXX) $(CXXFLAGS) $(CFLAGS) $< mains/lbfgsb.o mains/linpack.o $(LDFLAGS) -lnuSQuIDS -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $(CFLAGS) $< -o $@

mains/lbfgsb.o : ./inc/lbfgsb/lbfgsb.h ./inc/lbfgsb/lbfgsb.c
	$(CC) $(CFLAGS) ./inc/lbfgsb/lbfgsb.c -c -o ./mains/lbfgsb.o

mains/linpack.o : ./inc/lbfgsb/linpack.c
	$(CC) $(CFLAGS) ./inc/lbfgsb/linpack.c -c -o ./mains/linpack.o

.PHONY: clean
clean:
	rm -rf ./mains/*.exe ./bin/* ./mains/*.o

