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

add_library(alien_kernel_redistributor OBJECT
        converters/to_simple_csr_matrix.cc
        converters/to_simple_csr_matrix.h
        converters/to_simple_csr_vector.cc
        converters/to_simple_csr_vector.h
        converters/from_simple_csr_matrix.cc
        converters/from_simple_csr_matrix.h
        converters/from_simple_csr_vector.cc
        converters/from_simple_csr_vector.h
        RedistributorBackEnd.h
        Redistributor.cc
        RedistributorCommPlan.cc
        RedistributorCommPlan.h
        Redistributor.h
        RedistributorMatrix.cc
        RedistributorMatrix.h
        RedistributorVector.cc
        RedistributorVector.h
        Utils.h
        )

target_link_libraries(alien_kernel_redistributor PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_kernel_redistributor PUBLIC alien_utils alien_headers)

install(TARGETS alien_kernel_redistributor EXPORT ${ALIEN_EXPORT_TARGET})
