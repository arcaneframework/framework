/*
 * Copyright 2020 IFPEN-CEA
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

#include <alien/utils/Precomp.h>
#include <cmath>
#include <gtest/gtest.h>

#include <Environment.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/functional/BasicIndexManager.h>

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/handlers/block/BlockBuilder.h>

#include <alien/ref/AlienRefSemantic.h>

// Tests the default c'tor.
TEST(TestSchur, SchurEngine)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();

  auto cell1_global_size = 4 * comm_size;
  auto cell2_global_size = 4 * comm_size;

  Alien::UniqueArray<Alien::Int64> cell1_uid;
  Alien::UniqueArray<Alien::Integer> cell1_lid;
  Alien::UniqueArray<Alien::Int64> ghost_cell1_uid;
  Alien::UniqueArray<Alien::Integer> ghost_cell1_lid;
  Alien::UniqueArray<Alien::Integer> ghost_cell1_owner;
  Alien::UniqueArray<Alien::Int64> cell2_uid;
  Alien::UniqueArray<Alien::Integer> cell2_lid;
  Alien::UniqueArray<Alien::Int64> ghost_cell2_uid;
  Alien::UniqueArray<Alien::Integer> ghost_cell2_lid;
  Alien::UniqueArray<Alien::Integer> ghost_cell2_owner;

  auto cell1_offset = comm_rank * 4;
  auto cell1_local_size = 4;
  cell1_uid.reserve(cell1_local_size);
  cell1_lid.reserve(cell1_local_size);
  for (int i = 0; i < cell1_local_size; ++i) {
    cell1_uid.add(cell1_offset + i);
    cell1_lid.add(i);
  }

  auto cell2_offset = comm_rank * 4;
  auto cell2_local_size = 4;

  cell2_uid.reserve(cell2_local_size);
  cell2_lid.reserve(cell2_local_size);
  for (int i = 0; i < cell2_local_size; ++i) {
    cell2_uid.add(cell2_offset + i);
    cell2_lid.add(i);
  }

  if (comm_rank > 0) {
    ghost_cell1_uid.add(cell1_offset - 1);
    ghost_cell1_owner.add(comm_rank - 1);
    ghost_cell2_uid.add(cell2_offset - 1);
    ghost_cell2_owner.add(comm_rank - 1);
  }
  if (comm_rank < comm_size - 1) {
    ghost_cell1_uid.add(cell1_offset + cell1_local_size);
    ghost_cell1_owner.add(comm_rank + 1);
    ghost_cell2_uid.add(cell2_offset + cell2_local_size);
    ghost_cell2_owner.add(comm_rank + 1);
  }

  ghost_cell1_lid.resize(ghost_cell1_uid.size());
  Alien::AbstractFamily cell1_family(cell1_uid, ghost_cell1_uid, ghost_cell1_owner, AlienTest::Environment::parallelMng());
  cell1_family.uniqueIdToLocalId(ghost_cell1_lid, ghost_cell1_uid);

  ghost_cell2_lid.resize(ghost_cell2_uid.size());
  Alien::AbstractFamily cell2_family(cell2_uid, ghost_cell2_uid, ghost_cell2_owner, AlienTest::Environment::parallelMng());
  cell1_family.uniqueIdToLocalId(ghost_cell2_lid, ghost_cell2_uid);

  Alien::BasicIndexManager index_manager(AlienTest::Environment::parallelMng());

  auto indexSetU1 = index_manager.buildScalarIndexSet("U1", cell1_family);
  auto indexSetU2 = index_manager.buildScalarIndexSet("U2", cell2_family);

  index_manager.prepare();

  auto global_size = cell1_global_size + cell2_global_size;
  auto local_size = cell1_local_size + cell2_local_size;
  if (comm_rank == 0) {
    std::cout << "INFO(" << comm_rank << ") GLOBAL :" << global_size << " " << local_size
              << std::endl;
    std::cout << "INFO(" << comm_rank << ") CELL1   :" << cell1_global_size << " "
              << cell1_local_size << std::endl;
    std::cout << "INFO(" << comm_rank << ") CELL2   :" << cell2_global_size << " "
              << cell2_local_size << std::endl;
  }

  Alien::BlockBuilder block_builder(index_manager);

  auto allU1Index = indexSetU1.getOwnIndexes();
  auto allU2Index = indexSetU2.getOwnIndexes();
  block_builder[allU1Index] = 2;
  block_builder[allU2Index] = 3;

  Alien::VBlock vblock(block_builder.sizes());

  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(global_size, global_size, local_size, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(global_size, local_size, AlienTest::Environment::parallelMng());

  Alien::VBlockMatrix A(vblock, mdist);

  {
    trace_mng->info() << "CREATE MATRIX PROFILE";
    Alien::MatrixProfiler profiler(A);
    for (Integer i = 0; i < cell1_local_size; ++i) {
      Integer row1 = allU1Index[i];
      profiler.addMatrixEntry(row1, row1);
      trace_mng->info() << "ADD MATRIX ENTRY D1: " << row1 << " " << row1;

      Integer row2 = allU2Index[i];
      profiler.addMatrixEntry(row2, row2);
      trace_mng->info() << "ADD MATRIX ENTRY D2: " << row2 << " " << row2;

      profiler.addMatrixEntry(row1, row2);
      trace_mng->info() << "ADD MATRIX ENTRY D12: " << row1 << " " << row2;

      profiler.addMatrixEntry(row2, row1);
      trace_mng->info() << "ADD MATRIX ENTRY D21: " << row2 << " " << row1;

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU1Index[ghost_cell1_lid[0]];
          profiler.addMatrixEntry(row1, jcol);
          trace_mng->info() << "ADD MATRIX ENTRY OD1-1: " << row1 << " " << jcol;
        }
      }
      else {
        Integer jcol = allU1Index[i - 1];
        profiler.addMatrixEntry(row1, jcol);
        trace_mng->info() << "ADD MATRIX ENTRY OD1-1: " << row1 << " " << jcol;
      }

      if (i == cell1_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell1_lid[0] : ghost_cell1_lid[1];
          Integer jcol = allU1Index[lid];
          profiler.addMatrixEntry(row1, jcol);
          trace_mng->info() << "ADD MATRIX ENTRY OD1+1: " << row1 << " " << jcol << " " << comm_rank;
        }
      }
      else {
        Integer jcol = allU1Index[i + 1];
        profiler.addMatrixEntry(row1, jcol);
        trace_mng->info() << "ADD MATRIX ENTRY OD1+1: " << row1 << " " << jcol << " " << comm_rank;
      }

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU2Index[ghost_cell2_lid[0]];
          profiler.addMatrixEntry(row2, jcol);
          trace_mng->info() << "ADD MATRIX ENTRY OD1-1: " << row2 << " " << jcol;
        }
      }
      else {
        Integer jcol = allU2Index[i - 1];
        profiler.addMatrixEntry(row2, jcol);
        trace_mng->info() << "ADD MATRIX ENTRYOD2-1 : " << row2 << " " << jcol;
      }
      if (i == cell2_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell2_lid[0] : ghost_cell2_lid[1];
          Integer jcol = allU2Index[lid];
          profiler.addMatrixEntry(row2, jcol);
          trace_mng->info() << "ADD MATRIX ENTRY OD1+1: " << row2 << " " << jcol << " " << comm_rank;
        }
      }
      else {
        Integer jcol = allU2Index[i + 1];
        profiler.addMatrixEntry(row2, jcol);
        trace_mng->info() << "ADD MATRIX ENTRY OD2+1: " << row2 << " " << jcol << " " << comm_rank;
      }
    }
  }
  {
    trace_mng->info() << "FILL MATRIX";
    Alien::ProfiledVBlockMatrixBuilder builder(A, Alien::ProfiledVBlockMatrixBuilderOptions::eResetValues);
    for (Integer i = 0; i < cell1_local_size; ++i) {
      Integer row1 = allU1Index[i];
      auto row1_blk_size = vblock.size(row1);
      trace_mng->info() << "ROW1 index : " << row1 << " size : " << row1_blk_size;
      UniqueArray2<Real> diag1(row1_blk_size, row1_blk_size);
      diag1.fill(0);
      for (int k = 0; k < row1_blk_size; ++k)
        diag1[k][k] = 1.;
      builder(row1, row1) = diag1;

      Integer row2 = allU2Index[i];
      auto row2_blk_size = vblock.size(row2);
      trace_mng->info() << "ROW2 index : " << row2 << " size : " << row2_blk_size;

      UniqueArray2<Real> diag2(row2_blk_size, row2_blk_size);
      diag2.fill(0);
      for (int k = 0; k < row2_blk_size; ++k)
        diag2[k][k] = 1.;
      builder(row2, row2) = diag2;

      UniqueArray2<Real> off_diag12(row1_blk_size, row2_blk_size);
      off_diag12.fill(-0.012);
      builder(row1, row2) += off_diag12;

      UniqueArray2<Real> off_diag21(row2_blk_size, row1_blk_size);
      off_diag21.fill(-0.021);
      builder(row2, row1) += off_diag21;

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU1Index[ghost_cell1_lid[0]];
          auto col_blk_size = vblock.size(jcol);
          trace_mng->info() << "COL1-1 index : " << jcol << " size : " << col_blk_size;
          UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
          off_diag1.fill(-0.01);
          builder(row1, jcol) += off_diag1;
        }
      }
      else {
        Integer jcol = allU1Index[i - 1];
        auto col_blk_size = vblock.size(jcol);
        trace_mng->info() << "COL1-1 index : " << jcol << " size : " << col_blk_size;
        UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
        off_diag1.fill(-0.01);
        builder(row1, jcol) += off_diag1;
      }

      if (i == cell1_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell1_lid[0] : ghost_cell1_lid[1];
          Integer jcol = allU1Index[lid];
          auto col_blk_size = vblock.size(jcol);
          trace_mng->info() << "COL1+1 index : " << jcol << " size : " << col_blk_size;
          UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
          off_diag1.fill(-0.01);
          builder(row1, jcol) += off_diag1;
        }
      }
      else {
        Integer jcol = allU1Index[i + 1];
        auto col_blk_size = vblock.size(jcol);
        trace_mng->info() << "COL1+1 index : " << jcol << " size : " << col_blk_size;
        UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
        off_diag1.fill(-0.01);
        builder(row1, jcol) += off_diag1;
      }

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU2Index[ghost_cell2_lid[0]];
          auto col_blk_size = vblock.size(jcol);
          trace_mng->info() << "COL2-1 index : " << jcol << " size : " << col_blk_size;
          UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
          off_diag12.fill(-0.02);
          builder(row2, jcol) += off_diag2;
        }
      }
      else {
        Integer jcol = allU2Index[i - 1];
        auto col_blk_size = vblock.size(jcol);
        trace_mng->info() << "COL2-1 index : " << jcol << " size : " << col_blk_size;
        UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
        off_diag12.fill(-0.02);
        builder(row2, jcol) += off_diag2;
      }
      if (i == cell2_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell2_lid[0] : ghost_cell2_lid[1];
          Integer jcol = allU2Index[lid];
          auto col_blk_size = vblock.size(jcol);
          trace_mng->info() << "COL2+1 index : " << jcol << " size : " << col_blk_size;
          UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
          off_diag12.fill(-0.02);
          builder(row2, jcol) += off_diag2;
        }
      }
      else {
        Integer jcol = allU2Index[i + 1];
        auto col_blk_size = vblock.size(jcol);
        trace_mng->info() << "COL2+1 index : " << jcol << " size : " << col_blk_size;
        UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
        off_diag12.fill(-0.02);
        builder(row2, jcol) += off_diag2;
      }
    }
  }
}
