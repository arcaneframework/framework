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
 *  SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>

#include <Environment.h>

#include <alien/index_manager/functional/DefaultAbstractFamily.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/IAbstractFamily.h>
#include <random>

namespace
{
std::vector<size_t> generate_permutation(size_t length, int seed)
{
  std::vector<size_t> perm(length);
  auto i = 0;
  for (auto& p : perm) {
    p = i++;
  }

  std::shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));
  return perm;
}

template <typename T>
void apply_permutation(Arccore::ArrayView<T> array, const std::vector<size_t>& permutation)
{
  auto tmp_array = Arccore::UniqueArray<T>(array.constView());
  std::transform(permutation.begin(), permutation.end(), array.begin(), [&tmp_array](int id) { return tmp_array[id]; });
};

template <typename Family>
struct FamilyChecker
{
 public:
  Arccore::Int64 problem_size = 100;
  std::unique_ptr<Family> family = nullptr;

  explicit FamilyChecker(Arccore::MessagePassing::IMessagePassingMng* pm, bool shuffle = false)
  {
    Alien::UniqueArray<Arccore::Int64> uid;
    Alien::UniqueArray<Arccore::Int32> lid;
    Alien::UniqueArray<Arccore::Integer> owners;

    auto rank = pm->commRank();
    auto procs = pm->commSize();

    auto local_problem = problem_size / procs;

    auto start = local_problem * rank;

    if (rank == procs - 1) { // last processor
      local_problem = problem_size - start;
    }

    // Add local data
    for (auto i = 0; i < local_problem / 2; i++) {
      uid.add(i + start);
      lid.add(i);
      owners.add(rank);
    }

    for (auto i = 0; i < procs; i++) {
      if (i != rank) {
        auto offset = local_problem * i;
        uid.add(offset);
        lid.add(i + local_problem / 2);
        owners.add(i);
      }
    } // Add some ghosts

    // Resume local data
    for (auto i = local_problem / 2; i < local_problem; i++) {
      uid.add(i + start);
      lid.add(i + procs - 1);
      owners.add(rank);
    }

    assert(uid.size() == lid.size());
    assert(owners.size() == lid.size());

    if (shuffle) {
      auto perm = generate_permutation(lid.size(), 42);

      apply_permutation(uid.view(), perm);
      apply_permutation(lid.view(), perm);
      apply_permutation(owners.view(), perm);
    }

    family = std::make_unique<Family>(uid, owners, pm);
  }

  void simple_check()
  {
    auto sub_lid = family->allLocalIds();
    check(sub_lid.view());
  }

  void permuted_check(int seed)
  {
    auto sub_lid = Arccore::UniqueArray<Arccore::Int32>(family->allLocalIds().view());
    auto permutation = generate_permutation(sub_lid.size(), seed);
    apply_permutation(sub_lid.view(), permutation);
    check(sub_lid.view());
  }

  void check(Arccore::ConstArrayView<Arccore::Int32> lids)
  {
    // LIDs that we want to use
    auto local_size = lids.size();

    // UIDs of given LIDs
    auto test_uids = family->uids(lids);

    // Inverse lookup: from UIDs to LIDs
    Arccore::UniqueArray<Arccore::Int32> test_lid(local_size, 0);
    family->uniqueIdToLocalId(test_lid, test_uids.view());
    for (auto i = 0; i < lids.size(); i++) {
      ASSERT_EQ(test_lid[i], lids[i]);
    }
  }
};
} // namespace

TEST(TestFamilies, DefaultAbstractFamily)
{
  auto* pm = AlienTest::Environment::parallelMng();
  auto test_case = FamilyChecker<Alien::DefaultAbstractFamily>(pm);

  test_case.simple_check();
  test_case.permuted_check(37);
}

TEST(TestFamilies, DefaultAbstractFamilyShuffle)
{
  auto* pm = AlienTest::Environment::parallelMng();
  auto test_case = FamilyChecker<Alien::DefaultAbstractFamily>(pm, true);

  test_case.simple_check();
  test_case.permuted_check(37);
}

TEST(TestFamilies, AbstractItemFamily)
{
  auto* pm = AlienTest::Environment::parallelMng();
  auto test_case = FamilyChecker<Alien::AbstractItemFamily>(pm);

  test_case.simple_check();
  test_case.permuted_check(37);
}
