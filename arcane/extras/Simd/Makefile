SRC_FILES := Main.cc
SRC_PATH := $(shell pwd)/orig

CXX=opendist/gcc/gcc/4.9.3/bin/g++
CXX=opendist/gcc49/gcc/6.2/bin/g++

# Pour gcc, ajouter: -fopt-info-vec-all
# pour sortir les informations sur la vectorisation
OPT_FLAGS := -O2 -fstrength-reduce -fstrict-aliasing -ftree-vectorize
CXXFLAGS_AVX  := -DARCANE_HAS_AVX -mavx -march=corei7-avx -mavx2
CXXFLAGS_EMUL := -DARCANE_NO_SSE -DARCANE_NO_MIC
CXXFLAGS_SSE  :=
CXXFLAGS_AVX512 := -DARCANE_HAS_AVX512 -mavx512f
CXXFLAGS_NOVEC :=

ifeq (${USE_INTEL},yes)
CXX=icc-17.0.2.174/default/bin/icpc
OPT_FLAGS := -O2 -fstrict-aliasing -qopt-report-phase=vec -qopt-report=3
CXXFLAGS_AVX  := -DARCANE_HAS_AVX -xCORE-AVX2
CXXFLAGS_EMUL := -DARCANE_NO_SSE -DARCANE_NO_MIC
CXXFLAGS_SSE  :=
CXXFLAGS_AVX512 := -DARCANE_HAS_AVX512 -xMIC-AVX512
CXXFLAGS_NOVEC :=
endif

CXXFLAGS := -I. -I${SRC_PATH} -g -std=c++11 -Wall ${OPT_FLAGS} -lstdc++
LIBS=-lrt

all: novec sse avx avx512 emul

avx: test_avx.exe

sse: test_sse.exe

avx512: test_avx512.exe

emul: test_emul.exe

novec: test_novec.exe

ALL_DEPEND := bench/Wrapper.h  bench/Variables.h bench/Mesh.h \
	bench/HydroBenchBase.h bench/AlignedAllocator.h \
  ${SRC_PATH}/arcane/utils/ArcaneGlobal.h \
  ${SRC_PATH}/arcane/utils/Limits.h \
  ${SRC_PATH}/arcane/utils/Numeric.h \
  ${SRC_PATH}/arcane/utils/Real2.h \
  ${SRC_PATH}/arcane/utils/Real3.h \
  ${SRC_PATH}/arcane/utils/Simd.h \
  ${SRC_PATH}/arcane/utils/SimdAVX.h \
  ${SRC_PATH}/arcane/utils/SimdAVXGenerated.h \
  ${SRC_PATH}/arcane/utils/SimdCommon.h \
  ${SRC_PATH}/arcane/utils/SimdEMUL.h \
  ${SRC_PATH}/arcane/utils/SimdAVX512.h \
  ${SRC_PATH}/arcane/utils/SimdSSE.h \
  ${SRC_PATH}/arcane/utils/SimdEMULGenerated.h \
  ${SRC_PATH}/arcane/utils/SimdAVX512Generated.h \
  ${SRC_PATH}/arcane/utils/SimdSSEGenerated.h \
  ${SRC_PATH}/arcane/utils/SimdOperation.h \
  ${SRC_PATH}/arcane/utils/StdHeader.h \
  ${SRC_PATH}/arcane/SimdItem.h

ALL_OBJ := bench/Main.o bench/Mesh.o bench/NoVecHydroBench.o

test_sse.exe: bench/SimdHydroBench.cc ${ALL_DEPEND} ${ALL_OBJ} Makefile
	${CXX} ${CXXFLAGS} ${CXXFLAGS_SSE} ${ALL_OBJ} $< -o $@ ${LIBS}

test_avx.exe: bench/SimdHydroBench.cc ${ALL_DEPEND} ${ALL_OBJ} Makefile
	${CXX} ${CXXFLAGS} ${CXXFLAGS_AVX} ${ALL_OBJ} $< -o $@ ${LIBS}

test_avx512.exe: bench/SimdHydroBench.cc ${ALL_DEPEND} ${ALL_OBJ} Makefile
	${CXX} ${CXXFLAGS} ${CXXFLAGS_AVX512} ${ALL_OBJ} $< -o $@ ${LIBS}

test_emul.exe: bench/SimdHydroBench.cc ${ALL_DEPEND} ${ALL_OBJ} Makefile
	${CXX} ${CXXFLAGS} ${CXXFLAGS_EMUL} ${ALL_OBJ} $< -o $@ ${LIBS}

test_novec.exe:  bench/NoVecHydroBench2.cc ${ALL_DEPEND} ${ALL_OBJ} Makefile
	${CXX} ${CXXFLAGS} ${CXXFLAGS_NOVEC} ${ALL_OBJ} $< -o $@ ${LIBS}

bench/NoVecHydroBench.o: bench/NoVecHydroBench.cc ${ALL_DEPEND} Makefile
	${CXX} -c ${CXXFLAGS} ${CXXFLAGS_AVX512} $< -o $@

bench/Mesh.o: bench/Mesh.cc ${ALL_DEPEND} Makefile
	${CXX} -c ${CXXFLAGS} $< -o $@

bench/Main.o: bench/Main.cc ${ALL_DEPEND} Makefile
	${CXX} -c ${CXXFLAGS} $< -o $@

clean:
	-rm -f *.o bench/*.o *.exe

test: all
	-./test_avx.exe
	-./test_sse.exe
	-./test_avx512.exe
	-./test_emul.exe
	-./test_novec.exe
