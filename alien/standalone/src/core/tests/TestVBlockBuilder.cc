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
#include <alien/index_manager/functional/AbstractItemFamily.h>

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/handlers/block/BlockBuilder.h>

// Tests the default c'tor.
TEST(TestVBlockBuilder, VBlockFillingTest)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();

  auto ni = 4;
  auto nj = 2 * comm_size;
  auto node_global_size = ni * nj;
  auto cell_global_size = (ni - 1) * (nj - 1);

  Alien::UniqueArray<Alien::Int64> node_uid;
  Alien::UniqueArray<Alien::Integer> node_lid;
  Alien::UniqueArray<Alien::Int64> cell_uid;
  Alien::UniqueArray<Alien::Integer> cell_lid;

  auto node_local_size = 2 * ni;
  node_uid.reserve(node_local_size);
  node_lid.reserve(node_local_size);
  for (int i = 0; i < 2 * ni; ++i) {
    node_uid.add(comm_rank * 2 * ni + i);
    node_lid.add(i);
  }

  Alien::AbstractFamily node_family(
  node_uid, AlienTest::Environment::parallelMng());

  auto cell_local_size = 2 * (ni - 1);
  if (comm_rank == comm_size - 1)
    cell_local_size = ni - 1;

  cell_uid.reserve(cell_local_size);
  cell_lid.reserve(cell_local_size);
  for (int i = 0; i < cell_local_size; ++i) {
    cell_uid.add(comm_rank * 2 * (ni - 1) + i);
    cell_lid.add(i);
  }

  Alien::AbstractFamily cell_family(
  cell_uid, AlienTest::Environment::parallelMng());

  Alien::BasicIndexManager index_manager(AlienTest::Environment::parallelMng());

  trace_mng->info() << "BUILD SCALAR INDEX U";
  auto indexSetU = index_manager.buildScalarIndexSet("U", node_lid, node_family);

  trace_mng->info() << "BUILD SCALAR INDEX V";
  auto indexSetV = index_manager.buildScalarIndexSet("V", cell_lid, cell_family);

  index_manager.prepare();

  auto global_size = node_global_size + cell_global_size;
  auto local_size = node_local_size + cell_local_size;
  if (comm_rank == 0) {
    std::cout << "INFO(" << comm_rank << ") GLOBAL :" << global_size << " " << local_size
              << std::endl;
    std::cout << "INFO(" << comm_rank << ") NODE   :" << node_global_size << " "
              << node_local_size << std::endl;
    std::cout << "INFO(" << comm_rank << ") CELL   :" << cell_global_size << " "
              << cell_local_size << std::endl;
  }

  Alien::BlockBuilder block_builder(index_manager);

  //Arccore::UniqueArray<Integer> allUIndex = index_manager.getIndexes(indexSetU);
  //Arccore::UniqueArray<Integer> allVIndex = index_manager.getIndexes(indexSetV);
  auto u_indexes = indexSetU.getAllIndexes();
  auto v_indexes = indexSetV.getAllIndexes();
  trace_mng->info() << "SET BLOCK SIZE U";
  block_builder[u_indexes] = 2;

  trace_mng->info() << "SET BLOCK SIZE V";
  block_builder[v_indexes] = 3;

  Alien::VBlock vblock(block_builder.sizes());

  ASSERT_EQ(vblock.maxBlockSize(), 3);

  for (auto lid : u_indexes)
    ASSERT_EQ(vblock.size(lid), 2);

  for (auto lid : v_indexes)
    ASSERT_EQ(vblock.size(lid), 3);
}

TEST(TestVBlockBuilder, VBlockFillingTestWithGhost)
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
  Alien::UniqueArray<Alien::Integer> ghost_cell1_owner;
  Alien::UniqueArray<Alien::Int64> cell2_uid;
  Alien::UniqueArray<Alien::Integer> cell2_lid;
  Alien::UniqueArray<Alien::Int64> ghost_cell2_uid;
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

  Alien::AbstractFamily cell1_family(cell1_uid, ghost_cell1_uid, ghost_cell1_owner, AlienTest::Environment::parallelMng());

  Alien::AbstractFamily cell2_family(cell2_uid, ghost_cell2_uid, ghost_cell2_owner, AlienTest::Environment::parallelMng());

  Alien::BasicIndexManager index_manager(AlienTest::Environment::parallelMng());

  auto indexSetU1 = index_manager.buildScalarIndexSet("U1", cell1_lid, cell1_family);
  auto indexSetU2 = index_manager.buildScalarIndexSet("U2", cell2_lid, cell2_family);
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

  Arccore::UniqueArray<Integer> allU1Index = index_manager.getIndexes(indexSetU1);
  Arccore::UniqueArray<Integer> allU2Index = index_manager.getIndexes(indexSetU2);

  auto u1_indexes = indexSetU1.getOwnIndexes();
  auto u2_indexes = indexSetU2.getOwnIndexes();
  block_builder[u1_indexes] = 2;
  block_builder[u2_indexes] = 3;

  Alien::VBlock vblock(block_builder.sizes());

  ASSERT_EQ(vblock.maxBlockSize(), 3);

  for (auto lid : indexSetU1.getAllIndexes())
    ASSERT_EQ(vblock.size(lid), 2);

  for (auto lid : indexSetU2.getAllIndexes())
    ASSERT_EQ(vblock.size(lid), 3);
}
