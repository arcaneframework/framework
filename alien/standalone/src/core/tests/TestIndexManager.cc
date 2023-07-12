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

#include <gtest/gtest.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/functional/BasicIndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>

#include <Environment.h>

TEST(TestIndexManager, Constructor)
{
  // auto trace_mng = AlienTest::Environment::traceMng(); // not yet used

  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();
  auto ni = 4;
  auto nj = comm_size;
  auto global_size = ni * nj;

  Alien::UniqueArray<Alien::Int64> uid;
  Alien::UniqueArray<Alien::Integer> lid;

  auto local_size = ni;
  uid.reserve(local_size);
  lid.reserve(local_size);
  for (int i = 0; i < ni; ++i) {
    uid.add(comm_rank * ni + i);
    lid.add(i);
  }

  Alien::DefaultAbstractFamily family(uid, AlienTest::Environment::parallelMng());

  Alien::IndexManager index_manager(AlienTest::Environment::parallelMng());

  auto indexSetU = index_manager.buildScalarIndexSet("U", lid, family, 0);

  index_manager.prepare();

  ASSERT_EQ(index_manager.globalSize(), global_size);
  ASSERT_EQ(index_manager.localSize(), local_size);
}

TEST(TestIndexManager, ConstructorWithGhosts)
{
  auto trace_mng = AlienTest::Environment::traceMng(); // not yet used

  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();
  auto ni = 4;
  auto nj = comm_size;
  auto global_size = ni * nj;
  auto local_size = ni;

  Alien::UniqueArray<Alien::Int64> uids;
  Alien::UniqueArray<Alien::Integer> owners;

  uids.reserve(local_size + 2);
  owners.reserve(local_size + 2);
  for (int i = 0; i < ni; ++i) {
    uids.add(comm_rank * ni + i);
    owners.add(comm_rank);
  }

  int nb_ghost = 0;
  if (comm_size > 1) {
    if (comm_rank > 0) {
      uids.add(comm_rank * ni + -1);
      owners.add(comm_rank - 1);
      ++nb_ghost;
    }
    if (comm_rank < comm_size - 1) {
      uids.add(comm_rank * ni + ni);
      owners.add(comm_rank + 1);
      ++nb_ghost;
    }
  }

  Alien::AbstractItemFamily family(uids, owners, AlienTest::Environment::parallelMng());
  Alien::IndexManager index_manager(AlienTest::Environment::parallelMng(), trace_mng);

  auto indexSetU = index_manager.buildScalarIndexSet("U", family, 0);

  index_manager.prepare();

  ASSERT_EQ(index_manager.globalSize(), global_size);
  ASSERT_EQ(index_manager.localSize(), local_size);
}

TEST(TestIndexManager, ConstructorWithGhosts2)
{
  ALIEN_UNUSED_PARAM auto trace_mng = AlienTest::Environment::traceMng(); // not yet used

  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();
  auto ni = 4;
  auto nj = comm_size;
  auto global_size = ni * nj;
  auto local_size = ni;

  Alien::UniqueArray<Alien::Int64> uids;
  Alien::UniqueArray<Alien::Integer> lids;
  Alien::UniqueArray<Alien::Integer> owners;

  uids.reserve(local_size + 2);
  owners.reserve(local_size + 2);
  for (int i = 0; i < ni; ++i) {
    uids.add(comm_rank * ni + i);
    owners.add(comm_rank);
  }

  int nb_ghost = 0;
  if (comm_size > 1) {
    if (comm_rank > 0) {
      uids.add(comm_rank * ni + -1);
      owners.add(comm_rank - 1);
      ++nb_ghost;
    }
    if (comm_rank < comm_size - 1) {
      uids.add(comm_rank * ni + ni);
      owners.add(comm_rank + 1);
      ++nb_ghost;
    }
  }

  Alien::AbstractFamily family(uids, owners, AlienTest::Environment::parallelMng());

  Alien::BasicIndexManager index_manager(AlienTest::Environment::parallelMng());

  ALIEN_UNUSED_PARAM auto indexSetU = index_manager.buildScalarIndexSet("U", family);

  index_manager.prepare();

  ASSERT_EQ(index_manager.globalSize(), global_size);
  ASSERT_EQ(index_manager.localSize(), local_size);
}

TEST(TestIndexManager, MultiDofFamily)
{
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

  Alien::DefaultAbstractFamily node_family(
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

  Alien::DefaultAbstractFamily cell_family(
  cell_uid, AlienTest::Environment::parallelMng());

  Alien::IndexManager index_manager(AlienTest::Environment::parallelMng());

  auto indexSetU = index_manager.buildScalarIndexSet("U", node_lid, node_family, 0);
  auto indexSetV = index_manager.buildScalarIndexSet("V", cell_lid, cell_family, 1);

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
  ASSERT_EQ(index_manager.globalSize(), global_size);
  ASSERT_EQ(index_manager.localSize(), local_size);
}

TEST(TestIndexManager, DistributionConstructor)
{
  auto trace_mng = AlienTest::Environment::traceMng();
  auto comm_size = AlienTest::Environment::parallelMng()->commSize();
  auto comm_rank = AlienTest::Environment::parallelMng()->commRank();
  int ni = 4;
  int nj = comm_size;
  auto global_size = ni * nj;

  Alien::UniqueArray<Alien::Int64> uid;
  Alien::UniqueArray<Alien::Integer> lid;
  int local_size = ni;
  uid.reserve(local_size);
  lid.reserve(local_size);
  for (int i = 0; i < ni; ++i) {
    uid.add(comm_rank * ni + i);
    lid.add(i);
  }

  trace_mng->info() << "DefaultAbstractFamily" << comm_rank << " " << comm_size;

  Alien::DefaultAbstractFamily family(uid, AlienTest::Environment::parallelMng());

  Alien::IndexManager index_manager(AlienTest::Environment::parallelMng());

  auto indexSetU = index_manager.buildScalarIndexSet("U", lid, family, 0);

  trace_mng->info() << "index_manager::prepare" << comm_rank << " " << comm_size;
  index_manager.prepare();

  Alien::MatrixDistribution mdist = Alien::MatrixDistribution(
  global_size, global_size, local_size, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist = Alien::VectorDistribution(
  global_size, local_size, AlienTest::Environment::parallelMng());

  ASSERT_EQ(mdist.rowSpace().size(), global_size);
  ASSERT_EQ(vdist.space().size(), global_size);
}
