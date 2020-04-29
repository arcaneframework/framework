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

add_library(alien_semantic_ref
        include/alien/ref/AlienImportExport.h
        include/alien/ref/AlienRefSemantic.h
        include/alien/ref/AlienRefSemanticPrecomp.h
        src/block/BlockMatrix.cc
        include/alien/ref/data/block/BlockMatrix.h
        src/block/BlockVector.cc
        include/alien/ref/data/block/BlockVector.h
        src/block/VBlockMatrix.cc
        include/alien/ref/data/block/VBlockMatrix.h
        src/block/VBlockVector.cc
        include/alien/ref/data/block/VBlockVector.h
        src/scalar/Matrix.cc
        include/alien/ref/data/scalar/Matrix.h
        src/scalar/RedistributedMatrix.cc
        include/alien/ref/data/scalar/RedistributedMatrix.h
        src/scalar/RedistributedVector.cc
        include/alien/ref/data/scalar/RedistributedVector.h
        src/scalar/Vector.cc
        include/alien/ref/data/scalar/Vector.h
        src/functional/Ones.cc
        include/alien/ref/functional/Ones.h
        src/functional/Zeros.cc
        include/alien/ref/functional/Zeros.h
        include/alien/ref/handlers/block/ProfiledBlockMatrixBuilder.h
        include/alien/ref/handlers/profiler/MatrixProfiler.h
        include/alien/ref/handlers/scalar/DirectMatrixBuilder.h
        include/alien/ref/handlers/scalar/ProfiledMatrixBuilder.h
        src/stream/StreamMatrixBuilder.cc
        include/alien/ref/handlers/stream/StreamMatrixBuilder.h
        include/alien/ref/handlers/stream/StreamMatrixBuilderInserter.h
        include/alien/ref/handlers/stream/StreamMatrixBuilderInserterT.h
        include/alien/ref/handlers/stream/StreamMatrixBuilderT.h
        src/stream/StreamVBlockMatrixBuilder.cc
        include/alien/ref/handlers/stream/StreamVBlockMatrixBuilder.h
        include/alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserter.h
        include/alien/ref/handlers/stream/StreamVBlockMatrixBuilderInserterT.h
        include/alien/ref/handlers/stream/StreamVBlockMatrixBuilderT.h
        include/alien/ref/handlers/block/BlockVectorReader.h
        include/alien/ref/handlers/block/BlockVectorWriter.h
        include/alien/ref/handlers/block/VBlockVectorReader.h
        include/alien/ref/handlers/block/VBlockVectorWriter.h
        include/alien/ref/handlers/scalar/VectorReader.h
        include/alien/ref/handlers/scalar/VectorWriter.h
        include/alien/ref/import_export/SystemInfo.h
        src/import_export/SystemReader.cc
        include/alien/ref/import_export/SystemReader.h
        src/import_export/SystemWriter.cc
        include/alien/ref/import_export/SystemWriter.h
        src/import_export/HDF5Tools.h
        )

if (ALIEN_USE_XML)
    find_package(LibXml2 REQUIRED)
    target_link_libraries(alien_semantic_ref PRIVATE arccon::LibXml2)
    target_compile_definitions(alien_semantic_ref PRIVATE ALIEN_USE_LIBXML2)
endif ()

if (ALIEN_USE_HDF5)
    find_package(HDF5 REQUIRED)
    target_link_libraries(alien_semantic_ref PRIVATE arccon::HDF5)
    target_compile_definitions(alien_semantic_ref PRIVATE ALIEN_USE_HDF5)
endif ()


if (ALIEN_MV_EXPR)
    target_sources(alien_semantic_ref PRIVATE
            include/alien/ref/mv_expr/MVExpr.h
            )
    #TODO: check C++ features.
endif ()

target_link_libraries(alien_semantic_ref PUBLIC alien_core)

target_include_directories(alien_semantic_ref PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

generate_export_header(alien_semantic_ref
        BASE_NAME alien_semantic_ref
        EXPORT_MACRO_NAME ALIEN_REFSEMANTIC_EXPORT
        EXPORT_FILE_NAME include/alien/ref/AlienRefExport.h
        STATIC_DEFINE ${PROJECT_NAME}_BUILT_AS_STATIC
        )

install(TARGETS alien_semantic_ref
        EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(Alien::alien_semantic_ref ALIAS alien_semantic_ref)

install(DIRECTORY include/alien/ref
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alien/ref/AlienRefExport.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/ref
)
