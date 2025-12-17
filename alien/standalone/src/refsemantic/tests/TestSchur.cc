// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

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

#ifdef ALIEN_USE_EIGEN3
#include <alien/expression/schur/SchurOp.h>
#endif

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

  auto allU1Index = indexSetU1.getAllIndexes();
  auto allU2Index = indexSetU2.getAllIndexes();
  block_builder[allU1Index] = 2;
  block_builder[allU2Index] = 3;

  Alien::VBlock vblock(block_builder.sizes());

  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(global_size, global_size, local_size, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(global_size, local_size, AlienTest::Environment::parallelMng());

  Alien::VBlockMatrix A(vblock, mdist);
  Alien::VBlockVector b(vblock, vdist);
  Alien::VBlockVector x(vblock, vdist);

  {
    trace_mng->info() << "CREATE MATRIX PROFILE";
    Alien::MatrixProfiler profiler(A);
    for (Integer i = 0; i < cell1_local_size; ++i) {
      Integer row1 = allU1Index[i];
      profiler.addMatrixEntry(row1, row1);

      Integer row2 = allU2Index[i];
      profiler.addMatrixEntry(row2, row2);

      profiler.addMatrixEntry(row1, row2);

      profiler.addMatrixEntry(row2, row1);
      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU1Index[ghost_cell1_lid[0]];
          profiler.addMatrixEntry(row1, jcol);
        }
      }
      else {
        Integer jcol = allU1Index[i - 1];
        profiler.addMatrixEntry(row1, jcol);
      }

      if (i == cell1_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell1_lid[0] : ghost_cell1_lid[1];
          Integer jcol = allU1Index[lid];
          profiler.addMatrixEntry(row1, jcol);
        }
      }
      else {
        Integer jcol = allU1Index[i + 1];
        profiler.addMatrixEntry(row1, jcol);
      }

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU2Index[ghost_cell2_lid[0]];
          profiler.addMatrixEntry(row2, jcol);
        }
      }
      else {
        Integer jcol = allU2Index[i - 1];
        profiler.addMatrixEntry(row2, jcol);
      }
      if (i == cell2_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell2_lid[0] : ghost_cell2_lid[1];
          Integer jcol = allU2Index[lid];
          profiler.addMatrixEntry(row2, jcol);
        }
      }
      else {
        Integer jcol = allU2Index[i + 1];
        profiler.addMatrixEntry(row2, jcol);
      }
    }
  }
  {
    trace_mng->info() << "FILL VBLOCK MATRIX";
    Alien::ProfiledVBlockMatrixBuilder builder(A, Alien::ProfiledVBlockMatrixBuilderOptions::eResetValues);
    for (Integer i = 0; i < cell1_local_size; ++i) {
      Integer row1 = allU1Index[i];
      auto row1_blk_size = vblock.size(row1);
      UniqueArray2<Real> diag1(row1_blk_size, row1_blk_size);
      diag1.fill(-1. / (1 + i));
      for (int k = 0; k < row1_blk_size; ++k)
        diag1[k][k] = 1. + k;
      builder(row1, row1) = diag1;

      Integer row2 = allU2Index[i];
      auto row2_blk_size = vblock.size(row2);

      UniqueArray2<Real> diag2(row2_blk_size, row2_blk_size);
      diag2.fill(-2. / (1 + i));
      for (int k = 0; k < row2_blk_size; ++k)
        diag2[k][k] = 1. + k;
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
          UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
          off_diag1.fill(-0.01);
          builder(row1, jcol) += off_diag1;
        }
      }
      else {
        Integer jcol = allU1Index[i - 1];
        auto col_blk_size = vblock.size(jcol);
        UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
        off_diag1.fill(-0.01);
        builder(row1, jcol) += off_diag1;
      }

      if (i == cell1_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell1_lid[0] : ghost_cell1_lid[1];
          Integer jcol = allU1Index[lid];
          auto col_blk_size = vblock.size(jcol);
          UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
          off_diag1.fill(-0.01);
          builder(row1, jcol) += off_diag1;
        }
      }
      else {
        Integer jcol = allU1Index[i + 1];
        auto col_blk_size = vblock.size(jcol);
        UniqueArray2<Real> off_diag1(row1_blk_size, col_blk_size);
        off_diag1.fill(-0.01);
        builder(row1, jcol) += off_diag1;
      }

      if (i == 0) {
        if (comm_rank > 0) {
          Integer jcol = allU2Index[ghost_cell2_lid[0]];
          auto col_blk_size = vblock.size(jcol);
          UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
          off_diag12.fill(-0.02);
          builder(row2, jcol) += off_diag2;
        }
      }
      else {
        Integer jcol = allU2Index[i - 1];
        auto col_blk_size = vblock.size(jcol);
        UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
        off_diag12.fill(-0.02);
        builder(row2, jcol) += off_diag2;
      }
      if (i == cell2_local_size - 1) {
        if (comm_rank < comm_size - 1) {
          Integer lid = comm_rank == 0 ? ghost_cell2_lid[0] : ghost_cell2_lid[1];
          Integer jcol = allU2Index[lid];
          auto col_blk_size = vblock.size(jcol);
          UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
          off_diag12.fill(-0.02);
          builder(row2, jcol) += off_diag2;
        }
      }
      else {
        Integer jcol = allU2Index[i + 1];
        auto col_blk_size = vblock.size(jcol);
        UniqueArray2<Real> off_diag2(row2_blk_size, col_blk_size);
        off_diag12.fill(-0.02);
        builder(row2, jcol) += off_diag2;
      }
    }
  }

  //VBlock Vector Fill Step
  {
    trace_mng->info() << "FILL VBLOCK VECTOR";
    Alien::VBlockVectorWriter vb(b);
    for (Integer i = 0; i < cell1_local_size; ++i) {
      Integer row1 = allU1Index[i];
      auto row1_blk_size = vblock.size(row1);
      UniqueArray<Real> xb(row1_blk_size);
      xb.fill(1. / (i + 1));
      vb[row1].copy(xb);
    }
    for (Integer i = 0; i < cell2_local_size; ++i) {
      Integer row2 = allU2Index[i];
      auto row2_blk_size = vblock.size(row2);
      UniqueArray<Real> xb(row2_blk_size);
      xb.fill(2. / (i + 1));
      vb[row2].copy(xb);
    }
  }

#ifdef ALIEN_USE_EIGEN3
  Alien::SchurOp op(A, b);
  // REDUCE LINEAR SYSTEM DECLARATION
  {
    trace_mng->info() << "COMPUTE REDUCE STATIC BLOCK LINEAR SYSTEM WITH SCHUR OPERATOR";
    Integer block_size = 2;
    const Alien::Block block(block_size);
    Alien::BlockMatrix pA(block, mdist);
    Alien::BlockVector pb(block, vdist);
    Alien::BlockVector px(block, vdist);
    {
      Alien::BlockVectorWriter vpx(px);
      vpx = 1.;
    }

    trace_mng->info() << "COMPUTE PRIMARY SYSTEM";
    auto error = op.computePrimarySystem(pA, pb);

    trace_mng->info() << "COMPUTE SECONDARY SOLUTION FROM PRIMARY";
    auto error2 = op.computeSolutionFromPrimaryUnknowns(px, x);
  }

  {
    trace_mng->info() << "COMPUTE VBLOCK REDUCE LINEAR SYSTEM WITH SCHUR OPERATOR";
    Alien::BlockBuilder p_block_builder(index_manager);
    p_block_builder[allU1Index] = 1;
    p_block_builder[allU2Index] = 2;

    Alien::VBlock p_vblock(block_builder.sizes());
    Alien::VBlockMatrix pA(p_vblock, mdist);
    Alien::VBlockVector pb(p_vblock, vdist);
    Alien::VBlockVector px(p_vblock, vdist);
    {
      Alien::VBlockVectorWriter vpx(px);
      vpx = 1.;
    }

    trace_mng->info() << "COMPUTE PRIMARY SYSTEM";
    auto error = op.computePrimarySystem(pA, pb);

    trace_mng->info() << "COMPUTE SECONDARY SOLUTION FROM PRIMARY";
    auto error2 = op.computeSolutionFromPrimaryUnknowns(px, x);
  }
#endif
}

