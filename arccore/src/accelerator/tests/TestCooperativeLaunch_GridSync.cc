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

#include "arccore/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test performance de la synchronisation de grille
extern "C++" void
_testCooperativeLaunch_GridSync(RunQueue queue, Int32 nb_value, Int32 nb_loop, Int32 nb_loop2)
{
  double x = Platform::getRealTime();
  {
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      CooperativeWorkGroupLoopRange loop_range(nb_value);
      command << RUNCOMMAND_LAUNCH(iter, loop_range)
      {
        auto grid = iter.grid();
        for (Int32 j = 0; j < nb_loop2; ++j) {
          grid.barrier();
        }
      };
    }
  }
  double y = Platform::getRealTime();
  Real diff = (y - x) * 1000.0;
  std::cout << "** TotalCooperativeLaunch0 nb_value=" << nb_value
            << " nb_loop2=" << nb_loop2 << " time(ms)=" << diff << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
