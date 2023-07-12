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

set(alien_refsemantic_public_header
        alien/ref/AlienImportExport.h
        alien/ref/AlienRefSemantic.h
        alien/ref/AlienRefSemanticPrecomp.h
        alien/ref/data/block/BlockMatrix.h
        alien/ref/data/block/BlockVector.h
        alien/ref/data/block/VBlockMatrix.h
        alien/ref/data/block/VBlockVector.h
        alien/ref/data/scalar/Matrix.h
        alien/ref/data/scalar/RedistributedMatrix.h
        alien/ref/data/scalar/RedistributedVector.h
        alien/ref/data/scalar/Vector.h
        alien/ref/functional/Ones.h
        alien/ref/functional/Zeros.h
        alien/ref/handlers/block/ProfiledBlockMatrixBuilder.h
        alien/ref/handlers/profiler/MatrixProfiler.h
        alien/ref/handlers/scalar/DirectMatrixBuilder.h
        alien/ref/handlers/scalar/ProfiledMatrixBuilder.h
        alien/ref/handlers/stream/StreamMatrixBuilder.h
        alien/ref/handlers/stream/StreamMatrixBuilderInserter.h
        alien/ref/handlers/stream/StreamMatrixBuilderInserterT.h
        alien/ref/handlers/stream/StreamMatrixBuilderT.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilder.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserter.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserterT.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderT.h
        alien/ref/handlers/block/BlockVectorReader.h
        alien/ref/handlers/block/BlockVectorWriter.h
        alien/ref/handlers/block/VBlockVectorReader.h
        alien/ref/handlers/block/VBlockVectorWriter.h
        alien/ref/handlers/scalar/VectorReader.h
        alien/ref/handlers/scalar/VectorWriter.h
        alien/ref/import_export/SystemInfo.h
        alien/ref/import_export/SystemReader.h
        alien/ref/import_export/SystemWriter.h
        alien/ref/import_export/MatrixMarketSystemWriter.h
        alien/ref/import_export/HDF5Tools.h)

set(alien_semantic_ref_list
        alien/ref/AlienImportExport.h
        alien/ref/AlienRefSemantic.h
        alien/ref/AlienRefSemanticPrecomp.h
        alien/ref/data/block/BlockMatrix.cc
        alien/ref/data/block/BlockMatrix.h
        alien/ref/data/block/BlockVector.cc
        alien/ref/data/block/BlockVector.h
        alien/ref/data/block/VBlockMatrix.cc
        alien/ref/data/block/VBlockMatrix.h
        alien/ref/data/block/VBlockVector.cc
        alien/ref/data/block/VBlockVector.h
        alien/ref/data/scalar/Matrix.cc
        alien/ref/data/scalar/Matrix.h
        alien/ref/data/scalar/RedistributedMatrix.cc
        alien/ref/data/scalar/RedistributedMatrix.h
        alien/ref/data/scalar/RedistributedVector.cc
        alien/ref/data/scalar/RedistributedVector.h
        alien/ref/data/scalar/Vector.cc
        alien/ref/data/scalar/Vector.h
        alien/ref/functional/Ones.cc
        alien/ref/functional/Ones.h
        alien/ref/functional/Zeros.cc
        alien/ref/functional/Zeros.h
        alien/ref/handlers/block/ProfiledBlockMatrixBuilder.h
        alien/ref/handlers/profiler/MatrixProfiler.h
        alien/ref/handlers/scalar/DirectMatrixBuilder.h
        alien/ref/handlers/scalar/ProfiledMatrixBuilder.h
        alien/ref/handlers/stream/StreamMatrixBuilder.cc
        alien/ref/handlers/stream/StreamMatrixBuilder.h
        alien/ref/handlers/stream/StreamMatrixBuilderInserter.h
        alien/ref/handlers/stream/StreamMatrixBuilderInserterT.h
        alien/ref/handlers/stream/StreamMatrixBuilderT.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilder.cc
        alien/ref/handlers/stream/StreamVBlockMatrixBuilder.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserter.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserterT.h
        alien/ref/handlers/stream/StreamVBlockMatrixBuilderT.h
        alien/ref/handlers/block/BlockVectorReader.h
        alien/ref/handlers/block/BlockVectorWriter.h
        alien/ref/handlers/block/VBlockVectorReader.h
        alien/ref/handlers/block/VBlockVectorWriter.h
        alien/ref/handlers/scalar/VectorReader.h
        alien/ref/handlers/scalar/VectorWriter.h
        alien/ref/import_export/MatrixMarketSystemWriter.cc
        alien/ref/import_export/MatrixMarketSystemWriter.h
        alien/ref/import_export/SystemInfo.h
        alien/ref/import_export/SystemReader.cc
        alien/ref/import_export/SystemReader.h
        alien/ref/import_export/SystemWriter.cc
        alien/ref/import_export/SystemWriter.h
        alien/ref/import_export/HDF5Tools.h)

add_library(alien_semantic_ref ${alien_semantic_ref_list})

if (ALIEN_GENERATE_DOCUMENTATION)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    doxygen_add_docs(doxygen_api_docs_refsemantic
            ${alien_refsemantic_public_header}
            USE_STAMP_FILE
            COMMENT "Generate API-documents for Alien."
            )
endif (ALIEN_GENERATE_DOCUMENTATION)

if (ALIEN_USE_LIBXML2)
    find_package(LibXml2 REQUIRED)

    target_link_libraries(alien_semantic_ref PRIVATE arccon::LibXml2)
endif ()

if (ALIEN_USE_HDF5)
    find_package(HDF5 REQUIRED)

    target_link_libraries(alien_semantic_ref PRIVATE arccon::HDF5)
endif ()

target_sources(alien_semantic_ref PRIVATE
               alien/ref/mv_expr/MVExpr.h
              )

target_link_libraries(alien_semantic_ref PUBLIC alien_core)

target_include_directories(alien_semantic_ref PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

generate_export_header(alien_semantic_ref
        BASE_NAME alien_semantic_ref
        EXPORT_MACRO_NAME ALIEN_REFSEMANTIC_EXPORT
        EXPORT_FILE_NAME alien/ref/AlienRefExport.h
        STATIC_DEFINE ${PROJECT_NAME}_BUILT_AS_STATIC
        )

install(TARGETS alien_semantic_ref
        EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(Alien::alien_semantic_ref ALIAS alien_semantic_ref)

enable_sanitizers(alien_semantic_ref)

install(DIRECTORY alien/ref
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/
        FILES_MATCHING PATTERN "*.h"
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/alien/ref/AlienRefExport.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/ref
)

if (ALIEN_UNIT_TESTS)
    add_subdirectory(tests)
endif (ALIEN_UNIT_TESTS)