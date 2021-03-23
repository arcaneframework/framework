include ../../../../Makefile.config

CXX=g++ 

LIBS=-L${ARCANEDIR}/lib -Wl,--whole-archive -larcane_std -larcane -larcane_mpi -Wl,--no-whole-archive $(XERCES_LIB_PATH) ${MPI_LIBS} -lm
INCS=-I../../../../src -I..

AXL_EXE = ${ARCANEDIR}/bin/axl2cc.py

all : libPerfectGas.so

libPerfectGas.so: PerfectGasEOSService.o 
	$(CXX) -shared -o $@ $^

PerfectGasEOSService.o: PerfectGasEOSService.cc PerfectGasEOSService.h PerfectGasEOS_axl.h
	$(CXX) -c $(INCS) $< -o $@

PerfectGasEOS_axl.h : PerfectGasEOS.axl
	$(AXL_EXE) $<

clean::
	\rm -f *.o *.so *_axl.h
 	