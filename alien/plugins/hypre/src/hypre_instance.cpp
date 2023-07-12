/*
* Copyright 2021 IFPEN-CEA
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

#include "hypre_instance.h"

#include <HYPRE_utilities.h>

#ifdef ALIEN_HYPRE_DEVICE
#include <_hypre_utilities.h>
#endif // ALIEN_HYPRE_DEVICE

namespace
{
class AlienHypreInstance
{
 public:
  static void init_if_needed(MPI_Comm comm)
  {
    if (hypre_initialized)
      return;

    bind_devices(comm);

#ifdef HAVE_HYPRE_INIT
    HYPRE_Init();
#endif

    hypre_initialized = true;

    setup_devices();
  }

  static void bind_devices([[maybe_unused]] MPI_Comm comm)
  {
#ifdef ALIEN_HYPRE_DEVICE
    MPI_Comm shmcomm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    int shmrank;
    int shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_rank(shmcomm, &shmsize);
    MPI_Comm_free(&shmcomm);

    HYPRE_Int num_devices;
    hypre_GetDeviceCount(&num_devices);

    HYPRE_Int device = shmrank % num_devices;
    hypre_SetDevice(device, nullptr);
#endif // ALIEN_HYPRE_DEVICE
  }

  static void setup_devices()
  {
#ifdef ALIEN_HYPRE_DEVICE
    /* AMG in GPU memory (default) */
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    /* setup AMG on GPUs */
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
    /* use hypre's SpGEMM instead of cuSPARSE */
    HYPRE_SetSpGemmUseCusparse(false);
    /* use GPU RNG */
    HYPRE_SetUseGpuRand(true);
    HYPRE_PrintDeviceInfo();
#endif //ALIEN_HYPRE_DEVICE
  }

 private:
  static bool hypre_initialized;
};

bool AlienHypreInstance::hypre_initialized = false;
} // namespace

namespace Alien::Hypre
{

void hypre_init_if_needed(MPI_Comm comm)
{
  AlienHypreInstance::init_if_needed(comm);
}

} // namespace Alien::Hypre