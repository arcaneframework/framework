# NOTE: le package CUDA est obsol�te (CMake 3.11) car CUDA est maintenant directement
# support� comme langage par CMake
find_package(CUDA)
if (NOT TARGET arcane::CUDA)
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_aleph_cuda
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/cuda
  FILES ${ARCANE_SOURCES}
)

#TODO: trouver le nom g�n�r� par CMAke pour le chemin de AlephCuda.cu.o
set(ALEPH_CU ${CMAKE_CURRENT_SOURCE_DIR}/AlephCuda.cu)
set(ALEPH_CU_O CMakeFiles/arcane_aleph_cuda.dir/AlephCuda.cu.o)

add_custom_command(OUTPUT ${ALEPH_CU_O} COMMAND ${CUDA_BIN_NVCC} 
                   ARGS --compiler-bindir ${CMAKE_CXX_COMPILER} -I${Arccore_SOURCE_DIR}/src/base -I${Arccore_SOURCE_DIR}/src/collections -I${Arccore_SOURCE_DIR}/src/trace -I${Arccore_SOURCE_DIR}/src/concurrency -I${Arccore_SOURCE_DIR}/src/message_passing -I${ARCANE_SRC_PATH} -I${CMAKE_BINARY_DIR} -std=c++11 -shared -arch=sm_35 --compiler-options -fPIC -DARCANE_NOEXCEPT_FOR_NVCC -c -o ${ALEPH_CU_O} ${ALEPH_CU}
                   DEPENDS ${ALEPH_CU}
                   COMMENT " ######## NVCC ########\n\t########################")

target_sources(arcane_aleph_cuda PRIVATE ${ALEPH_CU_O})

target_link_libraries(arcane_aleph_cuda PRIVATE arcane_aleph arcane::CUDA)
target_link_libraries(arcane_aleph_cuda PUBLIC arcane_core)
arcane_register_library(arcane_aleph_cuda OPTIONAL)
