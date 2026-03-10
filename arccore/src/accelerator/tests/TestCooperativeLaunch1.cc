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
#include "arccore/common/accelerator/Memory.h"

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
_testCooperativeLaunch_GridSync(RunQueue queue, Int32 nb_value,
                                Int32 nb_loop1, Int32 nb_loop2);

extern "C++" Int64
_testCooperativeLaunch(RunQueue queue, SmallSpan<const Int64> c,
                       Int32 nb_value, Int32 nb_loop);
extern "C++" Int64
_testCooperativeLaunch2(RunQueue queue, SmallSpan<const Int64> c,
                        Int32 nb_value, Int32 nb_loop);

void _doTestCooperativeLaunch(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryResource::Device);
  Int32 nb_loop = 1000;
  Int32 nb_value = 1000000;
  Int64 expected_value = 1000004000000;

  nb_value = 10000000;
  expected_value = 100000040000000;
  nb_loop = 25;

  //nb_value = 50000000;
  //expected_value = 2500000200000000;
  //nb_loop = 20;

  if (!queue.isAcceleratorPolicy()) {
    if (arccoreIsDebug())
      nb_loop /= 20;
    else
      nb_loop /= 4;
    if (nb_loop == 0)
      nb_loop = 1;
  }
  //nb_loop = 1;

  std::cout << "Using accelerator policy name=" << queue.executionPolicy() << " nb_loop=" << nb_loop << "\n";

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
    Int32 nb_loop1 = 10000;
    Int32 nb_loop2 = 0; //
    _testCooperativeLaunch_GridSync(queue, nb_value, nb_loop1, nb_loop2);
  }
  for (Int32 k = 1; k < 5; ++k) {
    Int32 nb_loop1 = 1000;
    Int32 nb_loop2 = 9;
    _testCooperativeLaunch_GridSync(queue, nb_value, nb_loop1, nb_loop2);
  }
  for (Int32 k = 1; k < 5; ++k) {
    Int32 nb_loop1 = 100;
    Int32 nb_loop2 = 99;
    _testCooperativeLaunch_GridSync(queue, nb_value, nb_loop1, nb_loop2);
  }
  for (Int32 k = 1; k < 5; ++k) {
    Int32 nb_loop1 = 10;
    Int32 nb_loop2 = 999;
    _testCooperativeLaunch_GridSync(queue, nb_value, nb_loop1, nb_loop2);
  }

  for (Int32 k = 1; k < 5; ++k) {
    {
      Int64 v = _testCooperativeLaunch(queue, c, nb_value, nb_loop);
      Int64 v2 = v / nb_loop;
      ASSERT_EQ(v2, expected_value);
    }
  }

  for (Int32 k = 1; k < 5; ++k) {
    {
      Int64 v = _testCooperativeLaunch2(queue, c, nb_value, nb_loop);
      Int64 v2 = v / nb_loop;
      ASSERT_EQ(v2, expected_value);
    }
  }
}

ARCCORE_TEST_DO_TEST_ACCELERATOR(ArccoreAccelerator, TestCooperativeLaunch, _doTestCooperativeLaunch);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
