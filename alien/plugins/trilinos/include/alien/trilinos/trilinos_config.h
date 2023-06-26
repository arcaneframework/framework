/*
* Copyright 2022 IFPEN-CEA
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <alien/trilinos/TrilinosBackend.h>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Ifpack2_Factory.hpp>

namespace Alien::Trilinos
{
#ifdef ALIEN_PLUGIN_TRILINOS_OMP
using Node = Kokkos::Compat::KokkosOpenMPWrapperNode;
#elif ALIEN_PLUGIN_TRILINOS_CUDA
using Node = Kokkos::Compat::KokkosCudaWrapperNode;
#elif ALIEN_PLUGIN_TRILINOS_HIP
using Node = Kokkos::Compat::KokkosHIPWrapperNode;
#elif ALIEN_PLUGIN_TRILINOS_SERIAL
using Node = Kokkos::Compat::KokkosSerialWrapperNode;
#elif ALIEN_PLUGIN_TRILINOS_THREADS
using Node = Kokkos::Compat::KokkosThreadsWrapperNode;
#elif ALIEN_PLUGIN_TRILINOS_SYCL
using Node = Kokkos::Compat::KokkosSYCLWrapperNode;
#else
using Node = Kokkos::Compat::KokkosOpenMPWrapperNode;
#endif

using SC = double;
using LO = Tpetra::Map<>::local_ordinal_type;
using GO = Tpetra::Map<>::global_ordinal_type;
using MV = Tpetra::MultiVector<SC, LO, GO, Node>;
using OP = Tpetra::Operator<SC, LO, GO, Node>;
using map_type = Tpetra::Map<LO, GO, Node>;
using crs_matrix_type = Tpetra::CrsMatrix<SC, LO, GO, Node>;
using row_matrix_type = Tpetra::RowMatrix<SC, LO, GO, Node>;
using prec_type = Ifpack2::Preconditioner<SC, LO, GO, Node>;

void kokkos_node_verbose();
} // namespace Alien::Trilinos
