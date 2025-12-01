// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/GenericPartitioner.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/NumArrayViews.h"

#include "arcane/utils/Exception.h"

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

extern "C++" void _doPartition1(RunQueue queue, SmallSpan<Real> values, SmallSpan<Real> final_values);

TEST(ArcaneAccelerator, Partition)
{
  auto f = [] {
    _doInit();

    Runner runner(_defaultExecutionPolicy());
    RunQueue queue(makeQueue(runner));

    std::array<Real, 8> v1{ 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 2.3, 1.6 };
    std::array<Real, 8> r1{ 4.5, 3.5, 7.0, 4.2, 2.3, 1.6, -1.2, 1.3 };
    _doPartition1(queue, SmallSpan<Real>(v1), SmallSpan<Real>(r1));

    std::array<Real, 9> v2{ 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 2.3, 1.6, 1.1 };
    std::array<Real, 9> r2{ 4.5, 3.5, 7.0, 4.2, 2.3, 1.1, 1.6, -1.2, 1.3 };
    _doPartition1(queue, SmallSpan<Real>(v2), SmallSpan<Real>(r2));

    std::array<Real, 3> v3{ 1.3, -1.2, 0.5 };
    std::array<Real, 3> r3{ 0.5, -1.2, 1.3 };
    _doPartition1(queue, SmallSpan<Real>(v3), SmallSpan<Real>(r3));

    std::array<Real, 4> v4{ 3.5, 2.3, 4.5, 5.6 };
    std::array<Real, 4> r4{ 3.5, 2.3, 4.5, 5.6 };
    _doPartition1(queue, SmallSpan<Real>(v4), SmallSpan<Real>(r4));

    std::array<Real, 10> v5{ 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 2.3, 1.6, 1.1, 1.5 };
    std::array<Real, 10> r5{ 4.5, 3.5, 7.0, 4.2, 2.3, 1.5, 1.1, 1.6, -1.2, 1.3 };
    _doPartition1(queue, SmallSpan<Real>(v5), SmallSpan<Real>(r5));
  };
  return arcaneCallFunctionAndTerminateIfThrow(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
