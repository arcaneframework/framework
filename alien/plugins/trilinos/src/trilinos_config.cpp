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

#include <alien/trilinos/trilinos_config.h>

namespace Alien::Trilinos
{

#ifdef ALIEN_PLUGIN_TRILINOS_OMP
const std::string kokkos_node = "OpenMP";
#elif ALIEN_PLUGIN_TRILINOS_CUDA
const std::string kokkos_node = "Cuda";
#elif ALIEN_PLUGIN_TRILINOS_HIP
const std::string kokkos_node = "Hip";
#elif ALIEN_PLUGIN_TRILINOS_SERIAL
const std::string kokkos_node = "Serial";
#elif ALIEN_PLUGIN_TRILINOS_THREADS
const std::string kokkos_node = "Pthreads";
#elif ALIEN_PLUGIN_TRILINOS_SYCL
const std::string kokkos_node = "Sycl";
#else
const std::string kokkos_node = "Default_OpenMP";
#endif

void kokkos_node_verbose()
{
  std::cout << "Kokkos node : " << kokkos_node << std::endl;
}

} // namespace Alien::Trilinos
