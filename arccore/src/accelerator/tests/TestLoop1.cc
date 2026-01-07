// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

#include "./TestCommon.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_testLoopDirect(RunQueue queue, SmallSpan<const Int64> a, SmallSpan<const Int64> b,
                SmallSpan<Int64> c, Int32 nb_thread,
                Int32 nb_value, Int32 nb_part, Int32 nb_loop);

void _doTestLoop(bool use_accelerator)
{
  Accelerator::Initializer x(use_accelerator, 0);
  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryResource::Device);
  Int32 nb_loop = 200;
  Int32 nb_value = 5000000;

  Int32 nb_thread = 256;
  Int32 nb_part = 1;
  if (!queue.isAcceleratorPolicy()) {
    if (arccoreIsDebug()) {
      nb_value = 1000000;
      nb_loop = 1;
    }
    else
      nb_loop /= 4;
    if (nb_loop == 0)
      nb_loop = 1;
  }
  //nb_loop = 1;

  std::cout << "Using accelerator policy name=" << queue.executionPolicy() << "\n";
  std::cout << " nb_loop=" << nb_loop << " nb_value=" << nb_value << "\n";

  eMemoryResource mem = queue.memoryResource();
  NumArray<Int64, MDDim1> host_a(eMemoryResource::Host);
  NumArray<Int64, MDDim1> host_b(eMemoryResource::Host);
  NumArray<Int64, MDDim1> host_c(eMemoryResource::Host);
  host_a.resize(nb_value);
  host_b.resize(nb_value);
  host_c.resize(nb_value);
  {
    for (Int32 i = 0; i < nb_value; ++i) {
      host_a(i) = (i + 2);
      host_b(i) = (i + 3);
    };
  }

  NumArray<Int64, MDDim1> a(mem);
  NumArray<Int64, MDDim1> b(mem);
  NumArray<Int64, MDDim1> c(mem);
  a.copy(host_a);
  b.copy(host_b);
  c.resize(nb_value);
  for (Int32 k = 1; k < 5; ++k) {
    host_c.fill(0);
    c.copy(host_c);
    _testLoopDirect(queue, a, b, c, nb_thread, nb_value, nb_part, nb_loop);

    host_c.copy(c);
    for (Int32 i = 0; i < nb_value; ++i) {
      Int64 expected_value = host_a(i) + host_b(i);
      Int64 value = host_c(i);
      ASSERT_EQ(value, expected_value);
    }

    nb_part *= 2;
  }
}

ARCCORE_TEST_DO_TEST_ACCELERATOR(ArccoreAccelerator, TestLoop1, _doTestLoop);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
