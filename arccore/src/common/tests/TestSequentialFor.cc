// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/ForLoopRanges.h"

#include "arccore/common/SequentialFor.h"
#include "arccore/common/Array.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class LoopTester
{
 public:

  using IndexType = IndexType_;

 public:

  void doTest()
  {
    IndexType nb_dim1 = 20;
    SimpleForLoopRanges<1, IndexType> loop1(nb_dim1);
    UniqueArray<IndexType> sum_array;
    IndexType ref_sum = {};
    auto f = [&](MDIndex<1, IndexType> index) {
      IndexType i = index;
      sum_array.add(i);
      ref_sum += i;
    };
    arccoreSequentialFor(loop1, f);
    IndexType sum = {};
    for (IndexType x : sum_array) {
      sum += x;
    }
    ASSERT_EQ(ref_sum,sum);
  }
};

TEST(SequentialFor, Misc)
{
  LoopTester<Int32> int32_tester;
  int32_tester.doTest();

  LoopTester<Int64> int64_tester;
  int64_tester.doTest();

  LoopTester<size_t> size_t_tester;
  size_t_tester.doTest();
}
