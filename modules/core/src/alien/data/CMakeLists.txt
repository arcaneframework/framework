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

add_library(alien_data OBJECT
        CompositeMatrix.cc
        CompositeMatrix.h
        CompositeVector.cc
        CompositeVector.h
        IMatrix.h
        ISpace.h
        IVector.h
        Space.cc
        Space.h
        Universal.h
        Universe.cc
        UniverseDataBase.h
        Universe.h
        utils/ExtractionIndices.cc
        utils/ExtractionIndices.h
        utils/MatrixElement.h
        utils/Parameters.h
        utils/VectorElement.cc
        utils/VectorElement.h
        utils/VectorElementT.h
        )

target_include_directories(alien_data PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_include_directories(alien_data PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../..)
target_include_directories(alien_data PRIVATE ${PROJECT_BINARY_DIR})

target_link_libraries(alien_data PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_data PUBLIC alien_utils)

install(TARGETS alien_data EXPORT ${ALIEN_EXPORT_TARGET})