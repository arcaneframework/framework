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
#include "arccore/accelerator/RunCommandLoop.h"
#include "arccore/accelerator/Reduce.h"
#include "arccore/accelerator/internal/Initializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Int64
_testReduceDirect(RunQueue queue, SmallSpan<const Int64> c, Int32 nb_thread,
                  Int32 nb_value, Int32 nb_part, Int32 nb_loop, bool is_async)
{
  queue.setAsync(is_async);
  Int64 total_x = {};
  if ((nb_value % nb_part) != 0)
    ARCCORE_FATAL("{0} is not a multiple of {1}", nb_value, nb_part);
  Int32 nb_true_value = nb_value / nb_part;
  Int32 offset = nb_true_value;
  double x = Platform::getRealTime();
  {
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      ReducerSum2<Int64> reducer(command);
      command.addNbThreadPerBlock(nb_thread);
      command << RUNCOMMAND_LOOP1(iter, nb_true_value, reducer)
      {
        Int32 i = iter;
        for (Int32 k = 0; k < nb_part; ++k)
          reducer.combine(c_view[i + (offset * k)]);
      };
      Int64 tx = reducer.reducedValue();
      total_x += tx;
    }
  }
  double y = Platform::getRealTime();
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalReduceDirect=" << total_x << " async?=" << is_async
            << " nb_part=" << nb_part << " nb_value=" << nb_value
            << " nb_thread=" << nb_thread
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
  return total_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
