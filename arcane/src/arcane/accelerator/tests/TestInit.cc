// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

extern "C++" void arcaneRegisterDefaultAcceleratorRuntime();
extern "C++" eExecutionPolicy arcaneGetDefaultExecutionPolicy();

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

void _doTest1()
{
  _doInit();
  eExecutionPolicy exec_policy = _defaultExecutionPolicy();
  Runner runner(exec_policy);
  RunQueue queue(makeQueue(runner));
  ASSERT_TRUE(queue.executionPolicy() == exec_policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Accelerator, TestInit)
{
  _doTest1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
