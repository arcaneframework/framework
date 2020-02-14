# compilation des sources
if(TARGET trilinos)
if(ENABLE_CUDA)
set(CMAKE_CXX_COMPILER "$ENV{TRILINOS_ROOT}/bin/nvcc_wrapper")
endif()
add_subdirectory(src)
endif()

# on pourrait ajouter des tests etc
