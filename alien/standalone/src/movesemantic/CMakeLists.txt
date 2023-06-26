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

set(alien_semantic_public_header
        alien/move/handlers/block/BlockVectorReader.h
        alien/move/handlers/block/BlockVectorWriter.h
        alien/move/handlers/scalar/VectorReader.h
        alien/move/handlers/scalar/VectorWriter.h
        alien/move/AlienMoveSemantic.h
        alien/move/AlienMoveSemanticPrecomp.h
        alien/move/handlers/block/ProfiledBlockMatrixBuilder.h
        alien/move/handlers/scalar/DirectMatrixBuilder.h
        alien/move/handlers/scalar/MatrixProfiler.h
        alien/move/handlers/scalar/ProfiledMatrixBuilder.h
        alien/move/handlers/sub_matrix/Extraction.h
        alien/move/data/MatrixData.h
        alien/move/data/VectorData.h
        alien/move/handlers/scalar/DoKDirectMatrixBuilder.h
        alien/move/data/Redistribution.h)

add_library(alien_semantic_move
        ${alien_semantic_public_header}
        alien/move/handlers/sub_matrix/Extraction.cc
        alien/move/data/MatrixData.cc
        alien/move/data/VectorData.cc
        alien/move/impl/MatrixMarketReader.cc
        alien/move/data/Redistribution.cpp
        )

if (ALIEN_GENERATE_DOCUMENTATION)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    doxygen_add_docs(doxygen_api_docs_movesemantic
            ${alien_semantic_public_header}
            USE_STAMP_FILE
            COMMENT "Generate API-documents for Alien."
            )
endif (ALIEN_GENERATE_DOCUMENTATION)

target_link_libraries(alien_semantic_move PUBLIC alien_core)

target_include_directories(alien_semantic_move PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

generate_export_header(alien_semantic_move
        BASE_NAME alien_semantic_move
        EXPORT_MACRO_NAME ALIEN_MOVESEMANTIC_EXPORT
        EXPORT_FILE_NAME alien/move/AlienMoveSemanticExport.h
        STATIC_DEFINE ${PROJECT_NAME}_BUILT_AS_STATIC
        )

install(TARGETS alien_semantic_move
        EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(Alien::alien_semantic_move ALIAS alien_semantic_move)

enable_sanitizers(alien_semantic_move)

install(DIRECTORY alien/move
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        FILES_MATCHING PATTERN "*.h"
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/alien/move/AlienMoveSemanticExport.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/move/
)

if (ALIEN_UNIT_TESTS)
    add_subdirectory(tests)
endif (ALIEN_UNIT_TESTS)