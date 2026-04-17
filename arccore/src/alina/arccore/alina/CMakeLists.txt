include (srcs.cmake)

arccore_add_component_library(alina
  FILES ${SOURCES}
)

#arcane_add_library(ARCCORE_ALINA
#  INPUT_PATH ${Arcane_SOURCE_DIR}/src
#  RELATIVE_PATH arcane/alina
#  FILES ${ARCANE_SOURCES}
#)

target_compile_definitions(arccore_alina PRIVATE ARCANE_COMPONENT_arccore_alina)

target_link_libraries(arccore_alina PUBLIC Arcane::arcane_utils)
arcane_register_library(arccore_alina OPTIONAL)

if ((CMAKE_CXX_COMPILER_ID STREQUAL GNU) OR (CMAKE_CXX_COMPILER_ID STREQUAL Clang))
  target_compile_options(arccore_alina PUBLIC -Wno-conversion -Wno-float-conversion)
endif()

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if (WIN32)
  set(Boost_USE_STATIC_LIBS ON)
endif ()

find_package(Boost REQUIRED COMPONENTS
  headers
  program_options
)

if (Boost_FOUND)
  message(STATUS "Found boost")
  target_include_directories(arccore_alina SYSTEM PUBLIC
    ${Boost_INCLUDE_DIRS}
  )
  target_link_libraries(arccore_alina PUBLIC Boost::headers)
else()
  target_compile_definitions(arccore_alina INTERFACE ARCCORE_ALINA_NO_BOOST)
endif()

#----------------------------------------------------------------------------
# OpenMP

set (ARCCORE_ALINA_ENABLE_OMP TRUE)
if (ARCCORE_ALINA_ENABLE_OMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    target_compile_options(arccore_alina PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(arccore_alina PUBLIC
      $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
      $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
      $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
      )
  endif()
endif()

# Add Arccore MPI
# In the future, it will not be needed. Only 'arccore_message_passing' will be needed
target_link_libraries(arccore_alina PUBLIC Arccore::arccore_message_passing_mpi)

#----------------------------------------------------------------------------
# Eigen backend
#----------------------------------------------------------------------------
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)
  add_library(eigen_target INTERFACE)
  target_include_directories(eigen_target INTERFACE ${EIGEN_INCLUDE})
  target_compile_definitions(eigen_target INTERFACE ARCCORE_ALINA_HAVE_EIGEN)
endif()

#----------------------------------------------------------------------------
# Find CUDA
#----------------------------------------------------------------------------
if (ARCCORE_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (OPENMP_FOUND)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
  endif()

  add_library(cuda_target INTERFACE)
  target_link_libraries(cuda_target INTERFACE ${CUDA_cusparse_LIBRARY})
endif()

#----------------------------------------------------------------------------
# Find Pastix
#----------------------------------------------------------------------------
find_package(Metis  QUIET)

arcane_find_package(Parmetis)
if(Parmetis_FOUND)
  message(STATUS "Found Parmetis for Alina")
  target_compile_definitions(arccore_alina PUBLIC ARCCORE_ALINA_HAVE_PARMETIS ARCCORE_ALINA_HAVE_PARMETIS)
  target_link_libraries(arccore_alina PUBLIC arccon::Parmetis)
endif()

option(ARCCORE_ALINA_BUILD_TESTS "Build Alina tests" ON)
option(ARCCORE_ALINA_BUILD_SAMPLES "Build Alina samples" ON)

if (ARCCORE_ALINA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (ARCCORE_ALINA_BUILD_SAMPLES)
  add_subdirectory(samples)
  add_subdirectory(tutorial)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
