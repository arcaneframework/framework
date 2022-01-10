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
     *
     *           (I,J+1)
     *              |
     * (I-1,J) -- (I,J) -- (I+1,J)
     *              |
     *           (I,J-1)
     *
     * TUTORIAL : LINEAR SYSTEM mat.X=rhs DEFINITION
     * =========================================
     */
  int Nx = 10;
  int Ny = 10;
  // int space_size = Nx * Ny;

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
  Alien::UniqueArray<Arccore::Integer> owners;
  Alien::UniqueArray<LID> lid;
  std::map<UID, LID> uid2lid;
  int first_j = y_offset[comm_rank];
  int last_j = y_offset[comm_rank + 1];

  int index = 0;
  for (int j = first_j; j < last_j; ++j) {
    for (int i = 0; i < Nx; ++i) {
      int n_uid = node_uid(i, j);
      uid.add(n_uid);
      owners.add(comm_rank);
      lid.add(index);
      uid2lid[n_uid] = index;
      ++index;
    }
  }

  int nb_ghost = 0;
  if (comm_size > 1) {
    if (comm_rank > 0) {
      for (int i = 0; i < Nx; ++i) {
        int n_uid = node_uid(i, first_j - 1);
        uid.add(n_uid);
        owners.add(comm_rank - 1);
        lid.add(index);
        uid2lid[n_uid] = index;
        ++index;
        ++nb_ghost;
      }
    }
    if (comm_rank < comm_size - 1) {
      for (int i = 0; i < Nx; ++i) {
        int n_uid = node_uid(i, last_j + 1);
        uid.add(n_uid);
        owners.add(comm_rank + 1);
        lid.add(index);
        uid2lid[n_uid] = index;
        ++index;
        ++nb_ghost;
      }
    }
  }

  /*
     * DEFINITION of an abstract family of unknowns
     */
  Alien::DefaultAbstractFamily family(uid, owners, parallel_mng);

  Alien::IndexManager index_manager(parallel_mng);

  /*
     * Creation of a set of indexes
     */
  auto indexSetU = index_manager.buildScalarIndexSet("U", family, 0);

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

  auto matrix_dist =
  Alien::MatrixDistribution(global_size, global_size, local_size, parallel_mng);
  auto vector_dist = Alien::VectorDistribution(global_size, local_size, parallel_mng);

  trace_mng->info() << "MATRIX DISTRIBUTION INFO";
  trace_mng->info() << "GLOBAL ROW SIZE : " << matrix_dist.globalRowSize();
  trace_mng->info() << "LOCAL ROW SIZE  : " << matrix_dist.localRowSize();
  trace_mng->info() << "GLOBAL COL SIZE : " << matrix_dist.globalColSize();
  trace_mng->info() << "LOCAL COL SIZE  : " << matrix_dist.localColSize();

  trace_mng->info() << "VECTOR DISTRIBUTION INFO";
  trace_mng->info() << "GLOBAL SIZE : " << vector_dist.globalSize();
  trace_mng->info() << "LOCAL SIZE  : " << vector_dist.localSize();

  auto allUIndex = index_manager.getIndexes(indexSetU);

  /*
   *  Assemble matrix.
   */
  auto mat = Alien::Matrix(matrix_dist);

  /* Two passes */

  // PROFILE DEFINITION
  {
    Alien::MatrixProfiler profiler(mat);

    for (int j = first_j; j < last_j; ++j) {
      // BOUCLE SUIVANT AXE X
      for (int i = 0; i < Nx; ++i) {
        auto n_uid = node_uid(i, j);
        auto n_lid = uid2lid[n_uid];
        auto irow = allUIndex[n_lid];

        // DEFINE DIAGONAL
        profiler.addMatrixEntry(irow, irow);

        // OFF DIAG
        // lower
        if (j > 0) {
          auto off_uid = node_uid(i, j - 1);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            profiler.addMatrixEntry(irow, jcol);
        }
        // left
        if (i > 0) {
          auto off_uid = node_uid(i - 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            profiler.addMatrixEntry(irow, jcol);
        }
        // right
        if (i < Nx - 1) {
          auto off_uid = node_uid(i + 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            profiler.addMatrixEntry(irow, jcol);
        }
        // upper
        if (j < Ny - 1) {
          auto off_uid = node_uid(i, j + 1);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            profiler.addMatrixEntry(irow, jcol);
        }
      }
    }
  }

  // SECOND STEP : MATRIX FILLING STEP
  {
    Alien::ProfiledMatrixBuilder builder(mat, Alien::ProfiledMatrixOptions::eResetValues);
    // Loop on Y-axis
    for (int j = first_j; j < last_j; ++j) {
      // Loop on X-axis
      for (int i = 0; i < Nx; ++i) {
        auto n_uid = node_uid(i, j);
        auto n_lid = uid2lid[n_uid];
        auto irow = allUIndex[n_lid];

        // DIAGONAL
        builder(irow, irow) = 4;

        // OFF DIAG
        // lower
        if (j > 0) {
          auto off_uid = node_uid(i, j - 1);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // left
        if (i > 0) {
          auto off_uid = node_uid(i - 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // right
        if (i < Nx - 1) {
          auto off_uid = node_uid(i + 1, j);
          auto off_lid = uid2lid[off_uid];
          auto jcol = allUIndex[off_lid];
          if (jcol != -1)
            builder(irow, jcol) = -1;
        }
        // upper
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

  // FIXME: not always available.
  //  {
  //    Alien::SystemWriter writer("MatrixA.txt");
  //    writer.dump(mat);
  //  }

  /*
   * Build rhs vector
   */
  auto rhs = Alien::Vector(vector_dist);

  {
    Alien::VectorWriter writer(rhs);

    // Loop on Y-axis
    for (int j = first_j; j < last_j; ++j) {
      // Loop on X-axis
      for (int i = 0; i < Nx; ++i) {
        auto n_uid = node_uid(i, j);
        auto n_lid = uid2lid[n_uid];
        auto irow = allUIndex[n_lid];

        writer[irow] = 1. / (1. + i + j);
      }
    }
  }

  auto unknown = Alien::Vector(vector_dist);

  Environment::finalize();

  return 0;
}
