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
#include "arccore/common/accelerator/Memory.h"
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

void _doTestAllocation(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer init(use_accelerator, max_allowed_thread);
  Runner runner(init.executionPolicy());
  RunQueue queue(makeQueue(runner));
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryResource::Device);
  Int32 nb_value = 500 * 1000 * 128;
  if (!queue.isAcceleratorPolicy()) {
    nb_value /= 2;
  }
  Int32 nb_loop = 5;
  std::cout << "Using accelerator policy name=" << queue.executionPolicy() << "\n";
  std::cout << " nb_loop=" << nb_loop << " nb_value=" << nb_value << "\n";

  eMemoryResource mem = queue.memoryResource();
  NumArray<Int64, MDDim1> host_a(eMemoryResource::Host);
  host_a.resize(nb_value);

  // Test copie host -> device
  NumArray<Int64, MDDim1> device_a(mem);
  device_a.resize(nb_value / 2);
  device_a.fill(5, queue);
  ASSERT_EQ(device_a.extent0(), (nb_value / 2));

  device_a.copy(host_a);
  ASSERT_EQ(device_a.extent0(), host_a.extent0());

  // Test copie device -> UVM
  NumArray<Int64, MDDim1> device_b(device_a.to1DSpan());
  ASSERT_EQ(device_b.extent0(), device_a.extent0());
}

ARCCORE_TEST_DO_TEST_ACCELERATOR(ArccoreAccelerator, TestAllocation, _doTestAllocation);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
