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

void _doTestMemoryBandwidth(bool use_accelerator, Int32 max_allowed_thread)
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
  {
    for (Int32 i = 0; i < nb_value; ++i) {
      host_a(i) = (i + 2);
    };
  }

  NumArray<Int64, MDDim1> device_a(mem);
  device_a.copy(host_a);

  NumArray<Int64, MDDim1> device_b(mem);
  device_b.resize(nb_value);

  //! Teste la copie device/device
  MemoryCopyArgs copy_args(device_b.bytes(), device_a.bytes());
  double x = Platform::getRealTime();
  for (Int32 i = 0; i < nb_loop; ++i)
    queue.copyMemory(copy_args);
  queue.barrier();
  double y = Platform::getRealTime();
  // Il y a 1 tableau en lecture et 1 en écriture, soit 2 tableaux
  Int64 nb_byte = device_a.bytes().size() * nb_loop * 2;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalMemoryCopy Device/Device" << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
}

ARCCORE_TEST_DO_TEST_ACCELERATOR(ArccoreAccelerator, TestMemoryBandwidth, _doTestMemoryBandwidth);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
