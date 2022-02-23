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


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

find_package(hipSYCL CONFIG REQUIRED)
find_package(Boost COMPONENTS context)

add_library(alien_kernel_sycl OBJECT
        algebra/SYCLInternalLinearAlgebra.cc
        algebra/SYCLInternalLinearAlgebra.h
        algebra/SYCLLinearAlgebra.h
        algebra/SYCLBEllPackMatrixMult.h
        algebra/SYCLBEllPackMatrixMultT.h
        converters/SimpleCSRtoSYCLMatrixConverter.cc
        converters/SimpleCSRtoSYCLVectorConverter.cc
        converters/SYCLtoSimpleCSRVectorConverter.cc
        data/BEllPackStructInfo.h
        data/SendRecvOp.h
        data/SYCLBEllPackInternal.h
        data/SYCLBEllPackMatrix.cc
        data/SYCLBEllPackMatrix.h
        data/SYCLVector.cc
        data/SYCLVector.h
        SYCLBackEnd.h
        SYCLPrecomp.h
        )

target_compile_options(alien_kernel_sycl PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
add_sycl_to_target(TARGET alien_kernel_sycl 
                   SOURCES data/SYCLVector.cc 
                           data/SYCLBEllPackMatrix.cc
                           algebra/SYCLInternalLinearAlgebra.cc)

target_link_libraries(alien_kernel_sycl PUBLIC
                      Arccore::arccore_trace
                      Arccore::arccore_collections
                      Arccore::arccore_base
                      Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_kernel_sycl PUBLIC alien_utils alien_headers)

install(TARGETS alien_kernel_sycl EXPORT ${ALIEN_EXPORT_TARGET})


