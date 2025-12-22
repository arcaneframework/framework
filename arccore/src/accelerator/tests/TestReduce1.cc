// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/NumArray.h"

#include "arccore/accelerator/NumArrayViews.h"
#include "arccore/accelerator/RunCommandLoop.h"
#include "arccore/accelerator/Reduce.h"
#include "arccore/accelerator/internal/Initializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 _testReduceDirect(RunQueue queue, Int32 nb_value, Int32 nb_loop)
{
  eMemoryResource mem = queue.memoryResource();
  // Teste la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  // Définit 2 tableaux 'a' et 'b' et effectue leur initialisation.
  NumArray<Int64, MDDim1> c(mem);
  c.resize(nb_value);
  {
    auto command = makeCommand(queue);
    auto out_c = viewOut(command, c);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_c(i) = (i + 2) + (i + 3);
    };
  }

  Int64 total_x = {};
  double x = Platform::getRealTime();
  {
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      ReducerSum2<Int64> reducer(command);
      command << RUNCOMMAND_LOOP1(iter, nb_value, reducer)
      {
        reducer.combine(c_view[iter]);
      };
      Int64 tx = reducer.reducedValue();
      total_x += tx;
    }
  }
  double y = Platform::getRealTime();
  std::cout << "** TotalReduceDirect=" << total_x << " time=" << (y - x) << "\n";
  return total_x;
}

TEST(ArccoreAccelerator, TestReduceDirect)
{
  Accelerator::Initializer x(true, 0);
  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryResource::Device);
  Int32 nb_loop = 1000;
  if (arccoreIsDebug())
    nb_loop /= 20;
  Int32 nb_value = 1000000;
  Int64 v = _testReduceDirect(queue, nb_value, nb_loop);
  Int64 v2 = v /= nb_loop;
  Int64 expected_value = 1000004000000;
  std::cout << "V=" << v2 << "\n";
  ASSERT_EQ(v2, expected_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
