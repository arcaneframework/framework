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

extern "C++" void
_testLoopDirect(RunQueue queue, SmallSpan<const Int64> a, SmallSpan<const Int64> b,
                SmallSpan<Int64> c, Int32 nb_thread,
                Int32 nb_value, Int32 nb_part, Int32 nb_loop)
{
  if ((nb_value % nb_part) != 0)
    ARCCORE_FATAL("{0} is not a multiple of {1}", nb_value, nb_part);
  Int32 nb_true_value = nb_value / nb_part;
  Int32 offset = nb_true_value;
  double x = Platform::getRealTime();
  {
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      command.addNbThreadPerBlock(nb_thread);
      command << RUNCOMMAND_LOOP1(iter, nb_true_value)
      {
        Int32 i = iter;
        for (Int32 k = 0; k < nb_part; ++k) {
          Int32 z = i + (offset * k);
          c(z) = a(z) + b(z);
        }
      };
    }
  }
  double y = Platform::getRealTime();
  // Nombre d´octets transférés
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop * 3;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalLoopDirect "
            << " nb_part=" << nb_part << " nb_value=" << nb_value
            << " nb_thread=" << nb_thread
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
