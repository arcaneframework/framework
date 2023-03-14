# Copyright 2020 IFPEN-CEA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(alien/utils)
add_subdirectory(alien/data)
add_subdirectory(alien/kernels/composite)
add_subdirectory(alien/kernels/dok)
add_subdirectory(alien/kernels/redistributor)
add_subdirectory(alien/kernels/simple_csr)
add_subdirectory(alien/index_manager)

IF(ALIEN_USE_SYCL)
  find_package(hipSYCL CONFIG REQUIRED)
  add_subdirectory(alien/kernels/sycl)
endif()

set(alien_core_list
        alien/Alien.h
        alien/core/backend/BackEnd.h
        alien/core/backend/EigenSolver.h
        alien/core/backend/EigenSolverT.h
        alien/core/backend/IInternalEigenSolverT.h
        alien/core/backend/IInternalLinearAlgebraT.h
        alien/core/backend/IInternalLinearAlgebraExprT.h
        alien/core/backend/IInternalLinearSolverT.h
        alien/core/backend/IMatrixConverter.h
        alien/core/backend/IVectorConverter.h
        alien/core/backend/ISolverFabric.h
        alien/core/backend/LinearAlgebra.h
        alien/core/backend/LinearAlgebraT.h
        alien/core/backend/LinearAlgebraExpr.h
        alien/core/backend/LinearAlgebraExprT.h
        alien/core/backend/LinearSolver.h
        alien/core/backend/LinearSolverT.h
        alien/core/backend/MatrixConverterRegisterer.cc
        alien/core/backend/MatrixConverterRegisterer.h
        alien/core/backend/VectorConverterRegisterer.cc
        alien/core/backend/VectorConverterRegisterer.h
        alien/core/backend/SolverFabricRegisterer.cc
        alien/core/backend/SolverFabricRegisterer.h
        alien/core/backend/SolverFactory.h
        alien/core/block/Block.cc
        alien/core/block/Block.h
        alien/core/block/BlockSizeVector.cc
        alien/core/block/BlockSizeVector.h
        alien/core/block/ComputeBlockOffsets.h
        alien/core/block/IBlockBuilder.h
        alien/core/block/VBlock.cc
        alien/core/block/VBlock.h
        alien/core/block/VBlockOffsets.cc
        alien/core/block/VBlockOffsets.h
        alien/core/block/VBlockSizes.cc
        alien/core/block/VBlockSizes.h
        alien/core/impl/IMatrixImpl.cc
        alien/core/impl/IMatrixImpl.h
        alien/core/impl/IVectorImpl.cc
        alien/core/impl/IVectorImpl.h
        alien/core/impl/MultiMatrixImpl.cc
        alien/core/impl/MultiMatrixImpl.h
        alien/core/impl/MultiVectorImpl.cc
        alien/core/impl/MultiVectorImpl.h
        alien/core/utils/Partition.cc
        alien/core/utils/Partition.h
        alien/distribution/MatrixDistribution.cc
        alien/distribution/MatrixDistribution.h
        alien/distribution/VectorDistribution.cc
        alien/distribution/VectorDistribution.h
        alien/expression/normalization/NormalizeOpt.cc
        alien/expression/normalization/NormalizeOpt.h
        alien/expression/solver/IEigenSolver.h
        alien/expression/solver/ILinearAlgebra.h
        alien/expression/solver/ILinearSolver.h
        alien/expression/solver/SolverStat.cc
        alien/expression/solver/SolverStater.cc
        alien/expression/solver/SolverStater.h
        alien/expression/solver/SolverStat.h
        alien/functional/Cast.h
        alien/functional/Is.h
        alien/functional/NullMatrix.cc
        alien/functional/NullMatrix.h
        alien/functional/NullVector.cc
        alien/functional/NullVector.h
        alien/functional/UnsafeCast.h
        alien/handlers/block/ProfiledFixedBlockMatrixBuilder.cc
        alien/handlers/block/ProfiledFixedBlockMatrixBuilder.h
        alien/handlers/block/ProfiledVBlockMatrixBuilder.cc
        alien/handlers/block/ProfiledVBlockMatrixBuilder.h
        alien/handlers/block/BaseBlockVectorReader.cc
        alien/handlers/block/BaseBlockVectorReader.h
        alien/handlers/block/BaseBlockVectorWriter.cc
        alien/handlers/block/BaseBlockVectorWriter.h
        alien/handlers/block/BlockVectorReaderT.h
        alien/handlers/block/BlockVectorWriterT.h
        alien/handlers/block/BlockSizes.h
        alien/handlers/block/BlockSizes.cc
        alien/handlers/block/BlockBuilder.h
        alien/handlers/block/BlockBuilder.cc
        alien/handlers/profiler/BaseMatrixProfiler.cc
        alien/handlers/profiler/BaseMatrixProfiler.h
        alien/handlers/profiler/MatrixProfilerT.h
        alien/handlers/scalar/BaseDirectMatrixBuilder.cc
        alien/handlers/scalar/BaseDirectMatrixBuilder.h
        alien/handlers/scalar/BaseProfiledMatrixBuilder.cc
        alien/handlers/scalar/BaseProfiledMatrixBuilder.h
        alien/handlers/scalar/BaseVectorReader.cc
        alien/handlers/scalar/BaseVectorReader.h
        alien/handlers/scalar/BaseVectorWriter.cc
        alien/handlers/scalar/BaseVectorWriter.h
        alien/handlers/scalar/VectorReaderT.h
        alien/handlers/scalar/VectorWriterT.h
        alien/import_export/MatrixMarketSystemReader.h
        alien/import_export/MatrixMarketSystemReader.cc
        alien/import_export/Reader.h
        )

if (LibArchive_FOUND)
   list(APPEND alien_core_list
           alien/import_export/SuiteSparseArchiveSystemReader.cc
           alien/import_export/SuiteSparseArchiveSystemReader.h)
endif ()

add_library(alien_core ${alien_core_list})

target_link_libraries(alien_core PUBLIC alien_utils alien_data)
target_link_libraries(alien_core PUBLIC
                      alien_kernel_dok
                      alien_kernel_composite
                      alien_kernel_redistributor
                      alien_kernel_simplecsr)


target_link_libraries(alien_core PUBLIC alien_index_manager)

if(LibArchive_FOUND)
  target_link_libraries(alien_core PUBLIC archive)
endif ()

set_property(TARGET alien_core PROPERTY CXX_STANDARD 17)

message(STATUS "Using AVX Simd instructions ? -> ${ALIEN_WANT_AVX}")
message(STATUS "Using AVX2 Simd instructions ? -> ${ALIEN_WANT_AVX2}")
message(STATUS "Using AVX512 Simd instructions ? -> ${ALIEN_WANT_AVX512}")

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL Clang))
  if(ALIEN_WANT_AVX)
    target_compile_options(alien_core INTERFACE -mavx)
  endif()
  if(ALIEN_WANT_AVX2)
    target_compile_options(alien_core INTERFACE -mavx -mfma)
  endif()
  if(ALIEN_WANT_AVX512)
    target_compile_options(alien_core INTERFACE -mavx512f -mavx512cd)
  endif()
endif()

IF(ALIEN_USE_SYCL)
  target_link_libraries(alien_core PUBLIC alien_kernel_sycl)
  target_compile_options(alien_core PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
  add_sycl_to_target(TARGET alien_core)
ENDIF()


target_link_libraries(alien_core PUBLIC
                      Arccore::arccore_trace
                      Arccore::arccore_collections
                      Arccore::arccore_base
                      Arccore::arccore_message_passing_mpi)

find_package(Boost COMPONENTS program_options REQUIRED)

find_package(MPI REQUIRED)

target_link_libraries(alien_core PRIVATE MPI::MPI_CXX)
target_link_libraries(alien_core PUBLIC Boost::boost)

generate_export_header(alien_core
        EXPORT_MACRO_NAME ALIEN_EXPORT
        EXPORT_FILE_NAME alien/AlienExport.h)

target_link_libraries(alien_core PUBLIC alien_headers)

install(DIRECTORY alien
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
        )

install(TARGETS alien_core EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(TARGETS alien_headers EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/alien/AlienExport.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
)

add_library(Alien::alien_core ALIAS alien_core)

enable_sanitizers(alien_core)

if (ALIEN_UNIT_TESTS)
    add_subdirectory(tests)
endif (ALIEN_UNIT_TESTS)