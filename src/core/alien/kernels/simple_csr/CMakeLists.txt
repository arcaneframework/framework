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

add_library(alien_kernel_simplecsr OBJECT
        algebra/SimpleCSRInternalLinearAlgebra.cc
        algebra/SimpleCSRInternalLinearAlgebra.h
        algebra/SimpleCSRLinearAlgebra.h
        algebra/SimpleCSRMatrixMult.h
        algebra/SimpleCSRMatrixMultT.h
        CSRStructInfo.h
        DistStructInfo.cc
        DistStructInfo.h
        SendRecvOp.h
        SimpleCSRInternal.h
        SimpleCSRMatrix.cc
        SimpleCSRMatrix.h
        SimpleCSRVector.cc
        SimpleCSRVector.h
        redistribution/SimpleCSRVectorRedistributor.cc
        redistribution/SimpleCSRVectorRedistributor.h
        SimpleCSRBackEnd.h
        SimpleCSRPrecomp.h
        )

add_subdirectory(algebra)

target_include_directories(alien_kernel_simplecsr PRIVATE ../../..)
target_include_directories(alien_kernel_simplecsr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../..)
target_include_directories(alien_kernel_simplecsr PRIVATE ${PROJECT_BINARY_DIR})

find_package(BLAS REQUIRED)
target_link_libraries(alien_kernel_simplecsr PUBLIC ${BLAS_LIBRARIES})

target_link_libraries(alien_kernel_simplecsr PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_kernel_simplecsr PUBLIC alien_utils)

install(TARGETS alien_kernel_simplecsr EXPORT ${ALIEN_EXPORT_TARGET})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/algebra/CBLAS.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/kernels/simple_csr/algebra)

