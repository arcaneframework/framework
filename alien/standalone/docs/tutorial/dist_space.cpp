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

#include <mpi.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>

namespace Environment
{
void initialize(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
}

void finalize()
{
  MPI_Finalize();
}

Arccore::MessagePassing::IMessagePassingMng*
parallelMng()
{
  return Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
  MPI_COMM_WORLD);
}

Arccore::ITraceMng*
traceMng()
{
  return Arccore::arccoreCreateDefaultTraceMng();
}
} // namespace Environment

// Define index type for local ids
typedef Arccore::Integer LID;
// Define index type for global (unique) ids
typedef Arccore::Int64 UID;

int main(int argc, char** argv)
{
  /*
     * Example : Laplacian problem on a 2D square mesh of size NX x NY
     * Unknowns on nodes (i,j)
     * Use a 5-Points stencil
     */
  int Nx = 10;
  int Ny = 10;

  // INITIALIZE PARALLEL ENVIRONMENT
  Environment::initialize(argc, argv);

  auto parallel_mng = Environment::parallelMng();
  auto trace_mng = Environment::traceMng();

  auto comm_size = Environment::parallelMng()->commSize();
  auto comm_rank = Environment::parallelMng()->commRank();

  trace_mng->info() << "NB PROC = " << comm_size;
  trace_mng->info() << "RANK    = " << comm_rank;

  /*
     * MESH PARTITION ALONG Y AXIS
     *
     */
  int local_ny = Ny / comm_size;
  int r = Ny % comm_size;

  std::vector<int> y_offset(comm_size + 1);
  y_offset[0] = 0;
  for (int ip = 0; ip < r; ++ip)
    y_offset[ip + 1] = y_offset[ip] + local_ny + 1;

  for (int ip = r; ip < comm_size; ++ip)
    y_offset[ip + 1] = y_offset[ip] + local_ny;

  // Define a lambda function to compute node unique ids from the 2D (i,j) coordinates
  // (i,j) -> uid = node_uid(i,j)
  auto node_uid = [&](int i, int j) { return j * Ny + i; };

  /*
     * DEFINITION of Unknowns Unique Ids and  Local Ids
     */
  Alien::UniqueArray<UID> uid;
  Alien::UniqueArray<LID> lid;
  int first_j = y_offset[comm_rank];
  int last_j = y_offset[comm_rank + 1];

  int index = 0;
  for (int j = first_j; j < last_j; ++j) {
    for (int i = 0; i < Nx; ++i) {
      uid.add(node_uid(i, j));
      lid.add(index);
      ++index;
    }
  }

  /*
     * DEFINITION of an abstract family of unknowns
     */
  Alien::DefaultAbstractFamily family(uid, parallel_mng);

  Alien::IndexManager index_manager(parallel_mng);

  /*
     * Creation of a set of indexes
     */
  auto indexSetU = index_manager.buildScalarIndexSet("U", lid, family, 0);

  // Combine all index set and create Linear system index system
  index_manager.prepare();

  auto global_size = index_manager.globalSize();
  auto local_size = index_manager.localSize();

  trace_mng->info() << "GLOBAL SIZE : " << global_size;
  trace_mng->info() << "LOCAL SIZE  : " << local_size;

  /*
     * DEFINITION of
     * - Alien Space,
     * - matrix and vector distributions
     * to manage the distribution of indexes between all MPI processes
     */

  auto space = Alien::Space(global_size, "MySpace");

  auto mdist =
  Alien::MatrixDistribution(global_size, global_size, local_size, parallel_mng);
  auto vdist = Alien::VectorDistribution(global_size, local_size, parallel_mng);

  trace_mng->info() << "MATRIX DISTRIBUTION INFO";
  trace_mng->info() << "GLOBAL ROW SIZE : " << mdist.globalRowSize();
  trace_mng->info() << "LOCAL ROW SIZE  : " << mdist.localRowSize();
  trace_mng->info() << "GLOBAL COL SIZE : " << mdist.globalColSize();
  trace_mng->info() << "LOCAL COL SIZE  : " << mdist.localColSize();

  trace_mng->info() << "VECTOR DISTRIBUTION INFO";
  trace_mng->info() << "GLOBAL SIZE : " << vdist.globalSize();
  trace_mng->info() << "LOCAL SIZE  : " << vdist.localSize();

  Environment::finalize();

  return 0;
}
