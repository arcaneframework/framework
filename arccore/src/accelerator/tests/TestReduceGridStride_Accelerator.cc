// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"
#include "arccore/base/ForLoopRanges.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/NumArray.h"

#define ARCCORE_EXPERIMENTAL_GRID_STRIDE

namespace Arcane
{
class StridedLoop
: public SimpleForLoopRanges<1>
{
 public:

  using BaseClass = SimpleForLoopRanges<1>;
  using LoopIndexType = Arcane::MDIndex<1>;

 public:

  StridedLoop(Int32 nb_grid_stride, Int32 nb_value)
  : BaseClass(std::array<Int32, 1>{ (nb_value + (nb_grid_stride - 1)) / nb_grid_stride })
  , m_nb_grid_stride(nb_grid_stride)
  , m_nb_value(nb_value)
  {
  }
  constexpr Int32 nbGridStride() const { return m_nb_grid_stride; }
  constexpr Int32 nbTotalValue() const { return m_nb_value; }

 public:

  Int32 m_nb_grid_stride = 0;
  Int32 m_nb_value = 0;
};

ARCCORE_HOST_DEVICE MDIndex<1>
arcaneGetLoopIndexCudaHip([[maybe_unused]] StridedLoop loop_bounds, Int32 index)
{
  return MDIndex<1>(index);
}

} // namespace Arcane

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
_testReduceGridStride(RunQueue queue, SmallSpan<const Int64> c, Int32 nb_thread,
                      Int32 nb_value, Int32 nb_part, Int32 nb_loop, bool is_async)
{
  queue.setAsync(is_async);
  Int64 total_x = {};
  if ((nb_value % nb_part) != 0)
    ARCCORE_FATAL("{0} is not a multiple of {1}", nb_value, nb_part);
  if (!queue.isAcceleratorPolicy())
    nb_part = 1;
  double x = Platform::getRealTime();
  {
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      ReducerSum2<Int64> reducer(command);
      command.addNbThreadPerBlock(nb_thread);
      StridedLoop strided_loop(nb_part, nb_value);
      command << RUNCOMMAND_LOOP(iter, strided_loop, reducer)
      {
        reducer.combine(c_view[iter]);
      };
      Int64 tx = reducer.reducedValue();
      total_x += tx;
    }
  }
  double y = Platform::getRealTime();
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalReduceGridStride=" << total_x << " async?=" << is_async
            << " nb_part=" << nb_part << " nb_value=" << nb_value
            << " nb_thread=" << nb_thread
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
  return total_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
