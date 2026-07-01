// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/common/accelerator/Runner.h"

#include "arccore/accelerator/internal/Initializer.h"

#include "./TestCommon.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

extern "C++" void
_doTestLambda(RunQueue queue);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _doLambda(bool use_accelerator, Int32 max_allowed_thread)
{
  Accelerator::Initializer x(use_accelerator, max_allowed_thread);
  Runner runner(x.executionPolicy());
  RunQueue queue(makeQueue(runner));
  _doTestLambda(queue);
}

ARCCORE_TEST_DO_TEST_ACCELERATOR(ArccoreAccelerator, TestLambda, _doLambda);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
