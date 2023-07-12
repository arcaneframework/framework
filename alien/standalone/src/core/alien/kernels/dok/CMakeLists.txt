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

add_library(alien_kernel_dok OBJECT
        converters/from_simple_csr_matrix.cc
        converters/from_simple_csr_matrix.h
        converters/from_simple_csr_vector.cc
        converters/from_simple_csr_vector.h
        converters/to_simple_csr_matrix.cc
        converters/to_simple_csr_matrix.h
        converters/to_simple_csr_vector.cc
        converters/to_simple_csr_vector.h
        DoKBackEnd.h
        DoKDistributorComm.cc
        DoKDistributorComm.h
        DoKDistributor.h
        DoKDistributor.cc
        DoKLocalMatrixIndexer.cc
        DoKLocalMatrixIndexer.h
        DoKLocalMatrixT.h
        DoKMatrixT.h
        ILocalMatrixIndexer.h
        DoKReverseIndexer.cc
        DoKReverseIndexer.h
        IReverseIndexer.h
        BaseDoKDirectMatrixBuilder.cpp
        BaseDoKDirectMatrixBuilder.h
        DoKVector.h DoKVector.cc)

target_link_libraries(alien_kernel_dok PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_kernel_dok PUBLIC alien_utils alien_headers)
target_compile_features(alien_kernel_dok PUBLIC cxx_std_17)  # for std::optional

install(TARGETS alien_kernel_dok EXPORT ${ALIEN_EXPORT_TARGET})
