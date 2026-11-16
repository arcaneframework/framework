// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/FatalErrorException.h"

#include "arccore/accelerator/internal/Initializer.h"

#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ArccoreAccelerator, TestInit)
{
  Accelerator::Initializer x(true,0);

  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));

  // Check the device is the same between runner and queue.
  if (runner.deviceId() != queue.deviceId())
    ARCCORE_FATAL("Incoherent deviceId() between runner and queue");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
