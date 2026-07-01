// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/MDIndex.h"

#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/NumArray.h"

#include "arccore/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Extents> void
_fillArray(NumArray<Int64, Extents>& num_array, Int64 base_value)
{
  using ExtentIndexType = Extents::ExtentIndexType;
  ExtentIndexType nb_value = num_array.extent0();
  for (ExtentIndexType i = 0; i < nb_value; ++i) {
    num_array(i) = (i + base_value);
  }
}

void
_doTestLambda1(RunQueue queue)
{
  Int32 nb_value = 50000;

  std::cout << "Using accelerator policy name=" << queue.executionPolicy() << "\n";
  std::cout << " nb_value=" << nb_value << "\n";

  NumArray<Int64, MDDim1> a(nb_value);
  _fillArray(a, 3);

  NumArray<Int64, MDDim1> b(nb_value);

  // Test direct lambda launch
  {
    SmallSpan<const Int64> a_view(a.to1DSmallSpan());
    SmallSpan<Int64> b_view(a.to1DSmallSpan());
    auto lambda1 = [=] ARCCORE_HOST_DEVICE(Int32 index) {
      b_view[index] = a_view[index];
    };

    {
      std::cout << "Test lambda direct\n";
      RunCommand command = makeCommand(queue);
      auto range = makeLoopRanges(nb_value);
      Arcane::Accelerator::run(command, range, lambda1);
    }

    for (Int32 k = 0; k < nb_value; ++k) {
      ASSERT_EQ(a_view[k], b_view[k]) << " Index1=" << k;
    }
  }

  // Test direct mutable lambda launch
  {
    _fillArray(a, 5);

    ConstArrayView<Int64> a_view(nb_value, a.data());
    ArrayView<Int64> b_view(nb_value, b.data());
    auto mutable_lambda1 = [=] ARCCORE_HOST_DEVICE(Int32 index) mutable {
      b_view[index] = a_view[index];
    };

    {
      std::cout << "Test mutable lambda direct\n";
      RunCommand command = makeCommand(queue);
      auto range = makeLoopRanges(nb_value);
      Arcane::Accelerator::launchRunCommand(command, range, mutable_lambda1);
    }

    for (Int32 k = 0; k < nb_value; ++k) {
      ASSERT_EQ(a_view[k], b_view[k]) << " Index2=" << k;
    }
  }
}

// Check using a loop with a NumArray using 'Int64' as index type
void
_doTestLambda2(RunQueue queue)
{
  Int32 nb_value = 50000;

  std::cout << "Using accelerator policy name=" << queue.executionPolicy() << "\n";
  std::cout << " nb_value=" << nb_value << "\n";

  NumArray<Int64, MDDim1Ext<Int64>> a(nb_value);
  _fillArray(a, 3);

  NumArray<Int64, MDDim1Ext<Int64>> b(nb_value);

  // Test direct lambda launch
  {
    Span<const Int64> a_view(a.to1DSpan());
    Span<Int64> b_view(a.to1DSpan());
    auto lambda1 = [=] ARCCORE_HOST_DEVICE(Int64 index) {
      b_view[index] = a_view[index];
    };

    {
      std::cout << "Test lambda direct\n";
      RunCommand command = makeCommand(queue);
      SimpleForLoopRanges<1,Int64> loop_range(nb_value);
      launchRunCommand(command, loop_range, lambda1);
    }

    for (Int32 k = 0; k < nb_value; ++k) {
      ASSERT_EQ(a_view[k], b_view[k]) << " Index1=" << k;
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_doTestLambda(RunQueue queue)
{
  _doTestLambda1(queue);
  _doTestLambda2(queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
