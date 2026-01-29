// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void arcaneRegisterDefaultAcceleratorRuntime();
extern "C++" Arcane::Accelerator::eExecutionPolicy arcaneGetDefaultExecutionPolicy();

using namespace Arcane;
using namespace Arcane::Accelerator;

namespace
{
void _doInit()
{
  arcaneRegisterDefaultAcceleratorRuntime();
}
Arcane::Accelerator::eExecutionPolicy _defaultExecutionPolicy()
{
  return arcaneGetDefaultExecutionPolicy();
}
} // namespace

void _doReduce1()
{
  std::cout << "DO_REDUCE_1\n";
  Runner runner(_defaultExecutionPolicy());
  RunQueue queue(makeQueue(runner));
  {
    RunCommand command(makeCommand(queue));
    Int32 nb_iter = 96;
    ReducerSum2<Int64> reducer_sum(command);
    command << RUNCOMMAND_LOOP1(iter, nb_iter, reducer_sum)
    {
      auto [i] = iter();
      reducer_sum.combine(i + 1);
    };
    Int64 computed_sum = reducer_sum.reducedValue();
    std::cout << "NB_ITER=" << nb_iter << "\n";
    std::cout << "VALUE sum=" << computed_sum
              << "\n";
    Int64 expected_sum = (nb_iter * (nb_iter + 1)) / 2;
    std::cout << "VALUE sum=" << computed_sum
              << " expected=" << expected_sum
              << "\n";
    ASSERT_EQ(computed_sum, expected_sum);
  }
}

void _doReduce2()
{
  std::cout << "DO_REDUCE_2\n";
  Runner runner(_defaultExecutionPolicy());
  RunQueue queue(makeQueue(runner));
  {
    RunCommand command(makeCommand(queue));
    Int32 nb_iter = 4356;
    ReducerSum2<Int64> reducer_sum(command);
    ReducerMax2<Int64> reducer_max(command);
    ReducerMin2<Int64> reducer_min(command);
    command << ::Arcane::Accelerator::Impl::makeExtendedArrayBoundLoop(Arcane::SimpleForLoopRanges<1,Int32>(nb_iter),
                                                                       reducer_sum, reducer_max, reducer_min)
            << [=] ARCCORE_HOST_DEVICE(Arcane::ArrayIndex<1> iter, ReducerSum2<Int64> & reducer_sum, ReducerMax2<Int64> & reducer_max,
                                       ReducerMin2<Int64> & reducer_min) {
                 auto [i] = iter();
                 reducer_sum.combine(i + 1);
                 reducer_min.combine(i + 5);
                 reducer_max.combine(i - 5);
               };
    Int64 computed_sum = reducer_sum.reducedValue();
    Int64 computed_min = reducer_min.reducedValue();
    Int64 computed_max = reducer_max.reducedValue();
    std::cout << "VALUE sum=" << computed_sum
              << " min=" << computed_min
              << " max=" << computed_max
              << "\n";

    Int64 expected_sum = (nb_iter * (nb_iter + 1)) / 2;

    std::cout << "VALUE sum=" << computed_sum
              << " expected=" << expected_sum
              << "\n";
    std::cout << "MIN/MAX VALUE min=" << computed_min
              << " max=" << computed_max
              << "\n";

    ASSERT_EQ(computed_sum, expected_sum);
    ASSERT_EQ(computed_min, 5);
    ASSERT_EQ(computed_max, (nb_iter - 6));
  }
}

TEST(ArcaneAccelerator, Reduce)
{
  _doInit();

  bool is_copyable = std::is_trivially_copyable_v<ReducerSum2<Int64>>;
  std::cout << "IS_REDUCE_COPYABLE=" << is_copyable << "\n";
  //ASSERT_TRUE(is_copyable);

  _doReduce1();
  _doReduce2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
