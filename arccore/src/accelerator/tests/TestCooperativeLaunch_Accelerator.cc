// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/NumArray.h"

#include "arccore/accelerator/NumArrayViews.h"

#include "arccore/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Int64
_testCooperativeLaunch(RunQueue queue, SmallSpan<const Int64> c, Int32 nb_thread,
                       Int32 nb_value, Int32 nb_part, Int32 nb_loop, bool is_async)
{
  queue.setAsync(is_async);
  Int64 total_x = {};
  if ((nb_value % nb_part) != 0)
    ARCCORE_FATAL("{0} is not a multiple of {1}", nb_value, nb_part);
  Int32 nb_group = 10;
  Int32 group_size = 128;
  double x = Platform::getRealTime();
  {
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      command.addNbThreadPerBlock(nb_thread);
      auto loop_range = makeCooperativeWorkGroupLoopRange(command, nb_group, group_size);
      command << RUNCOMMAND_LAUNCH(iter, loop_range)
      {
        auto x = iter.group();
        //printf("X=%d\n", x.gridDim());
        x.gridBarrier();
        //Int32 i = iter;
      };
    }
  }
  double y = Platform::getRealTime();
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalCooperativeLaunch=" << total_x << " async?=" << is_async
            << " nb_part=" << nb_part << " nb_value=" << nb_value
            << " nb_thread=" << nb_thread
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
  return total_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
