include ../../../../Makefile.config

CXX=g++ 

LIBS=-L${ARCANEDIR}/lib -Wl,--whole-archive -larcane_std -larcane -larcane_mpi -Wl,--no-whole-archive $(XERCES_LIB_PATH) ${MPI_LIBS} -lm
INCS=-I../../../../src -I..

AXL_EXE = ${ARCANEDIR}/bin/axl2cc.py

all : libStiffenedGas.so

libStiffenedGas.so: StiffenedGasEOSService.o 
	$(CXX) -shared -o $@ $^

StiffenedGasEOSService.o: StiffenedGasEOSService.cc StiffenedGasEOSService.h StiffenedGasEOS_axl.h
	$(CXX) -c $(INCS) $< -o $@

StiffenedGasEOS_axl.h : StiffenedGasEOS.axl
	$(AXL_EXE) $<

clean::
	\rm -f *.o *.so *_axl.h
