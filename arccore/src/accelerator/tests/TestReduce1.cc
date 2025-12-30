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

extern "C++" Int64
_testReduceDirect(RunQueue queue, SmallSpan<const Int64> c, Int32 nb_thread, Int32 nb_value, Int32 nb_part, Int32 nb_loop);

TEST(ArccoreAccelerator, TestReduceDirect)
{
  Accelerator::Initializer x(true, 0);
  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryResource::Device);
  Int32 nb_loop = 1500;
  if (arccoreIsDebug())
    nb_loop /= 20;
  //nb_loop = 1;
  Int32 nb_value = 1000000;
  nb_value = 10000000;
  Int64 expected_value = 1000004000000;
  expected_value = 100000040000000;
  Int32 nb_thread = 256;
  Int32 nb_part = 1;

  eMemoryResource mem = queue.memoryResource();
  NumArray<Int64, MDDim1> host_c(eMemoryResource::Host);
  host_c.resize(nb_value);
  {
    for (Int32 i = 0; i < nb_value; ++i) {
      host_c(i) = (i + 2) + (i + 3);
    };
  }

  NumArray<Int64, MDDim1> c(mem);
  c.copy(host_c);

  for (Int32 k = 1; k < 5; ++k) {
    {
      // Test avec RunQueue synchrone
      std::cout << "Test Sync nb_part=" << nb_part << "\n";
      Int64 v = _testReduceDirect(queue, c, nb_thread, nb_value, nb_part, nb_loop);
      Int64 v2 = v / nb_loop;
      std::cout << "V=" << v2 << "\n";
      ASSERT_EQ(v2, expected_value);
    }
    {
      // Test avec RunQueue asynchrone
      std::cout << "Test Asynchronous nb_part=" << nb_part << "\n";
      queue.setAsync(true);
      Int64 v = _testReduceDirect(queue, c, nb_thread, nb_value, nb_part, nb_loop);
      Int64 v2 = v / nb_loop;
      std::cout << "V=" << v2 << "\n";
      ASSERT_EQ(v2, expected_value);
    }
    nb_part *= 2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
