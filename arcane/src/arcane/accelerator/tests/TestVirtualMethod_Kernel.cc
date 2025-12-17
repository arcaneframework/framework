// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include "TestVirtualMethod.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

extern "C++" void
_doCallTestVirtualMethod1(RunQueue& queue, NumArray<Int32, MDDim1>& compute_array, BaseTestClass* base_instance)
{
  // Applique une commande prenant en argument le pointeur sur la classe de base.
  const Int32 nb_item = compute_array.dim1Size();
  {
    RunCommand command(makeCommand(queue));
    auto in_out_array = compute_array.to1DSpan();
    command << RUNCOMMAND_LOOP1(iter, nb_item)
    {
      auto [i] = iter();
      in_out_array[i] = base_instance->apply(i, i);
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
