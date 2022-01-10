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
#include <map>
#include <mpi.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>

#include <alien/ref/AlienRefSemantic.h>

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
     * Example : LAPLACIAN PROBLEM on a 2D square mesh of size NX x NY
     * Unknowns on nodes (i,j)
     * Use a 5-Points stencil
     *
     *           (I,J+1)
     *              |
     * (I-1,J) -- (I,J) -- (I+1,J)
     *              |
     *           (I,J-1)
     *
     *
     * TUTORIAL : How to build the Laplacian matrix
     * ============================================
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

  // Define a lambda function to compute node uids from the 2D (i,j) coordinates
  // (i,j) -> uid = node_uid(i,j)
  auto node_uid = [&](int i, int j) { return j * Nx + i; };

  /*
     * DEFINITION of Unknowns Unique Ids and  Local Ids
     */
  Alien::UniqueArray<UID> uid;
  Alien::UniqueArray<LID> lid;
  std::map<UID, LID> uid2lid;
  int first_j = y_offset[comm_rank];
  int last_j = y_offset[comm_rank + 1];

  int index = 0;
  for (int j = first_j; j < last_j; ++j) {
    for (int i = 0; i < Nx; ++i) {
      int n_uid = node_uid(i, j);
      uid.add(n_uid);
      lid.add(index);
      uid2lid[n_uid] = index;
      ++index;
    }
  }

  /*
     * CREATION of an abstract family of unknowns
     */
  Alien::DefaultAbstractFamily family(uid, parallel_mng);

  /*
     * CREATION of an index manager
     */
  Alien::IndexManager index_manager(parallel_mng);

  /*
     * Creation of a set of indexes
     */
  auto indexSetU = index_manager.buildScalarIndexSet("U", lid, family, 0);

  // Combine all index sets to create a global distributed index system
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

  auto allUIndex = index_manager.getIndexes(indexSetU);
  /*
     * CONSTRUCTION DE LA MATRIX
     */
  auto A = Alien::Matrix(mdist);

  /* REMPLISSAGE EN UNE PASSE */

  trace_mng->info() << "REMPLISSAGE DIRECTE EN UNE PASSE";

  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    auto builder = Alien::DirectMatrixBuilder(
    A, tag, Alien::DirectMatrixOptions::SymmetricFlag::eUnSymmetric);

    // RESERVE 5 elements non nulles par ligne
    builder.reserve(5);
    builder.allocate();

    // BOUCLE SUIVANT AXE Y
    for (int j = first_j; j < last_j; ++j) {
      // BOUCLE SUIVANT AXE X
      for (int i = 0; i < Nx; ++i) {
        auto n_uid = node_uid(i, j);
        auto n_lid = uid2lid[n_uid];
        auto irow = allUIndex[n_lid];

        // REMPLISSAGE DIAGONAL
        builder(irow, irow) = 4;

        // Remplissage OFF DIAG
        // En Bas
        if (j > 0) {
          auto off_uid = node_uid(i, j - 1);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // A Gauche
        if (i > 0) {
          auto off_uid = node_uid(i - 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // A droite
        if (i < Nx - 1) {
          auto off_uid = node_uid(i + 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // En haut
        if (j < Ny - 1) {
          auto off_uid = node_uid(i, j + 1);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
      }
    }
  }

  Environment::finalize();

  return 0;
}
