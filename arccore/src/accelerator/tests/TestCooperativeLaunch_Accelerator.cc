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

#include "arccore/accelerator/RunCommandLaunch.h"

#include "arccore/accelerator/Reduce.h"
#include "arccore/accelerator/LocalMemory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Int64
_testCooperativeLaunch(RunQueue queue, SmallSpan<const Int64> c, Int32 nb_thread,
                       Int32 nb_value, Int32 nb_part, Int32 nb_loop)
{
  Int64 total_x = 0;
  if ((nb_value % nb_part) != 0)
    ARCCORE_FATAL("{0} is not a multiple of {1}", nb_value, nb_part);
  //Int32 nb_group = 15 * 1200;
  //Int32 group_size = 128;
  double x = Platform::getRealTime();
  // Valeurs partielles par bloc.
  // Doit être dimensionné au nombre maximum de blocs possibles
  NumArray<Int64, MDDim1> by_block_partial_sum(queue.memoryResource());
  by_block_partial_sum.resize(2048);
  // Pour récupèrer le résultat de la réduction.
  NumArray<Int64, MDDim1> reduce_result(eMemoryResource::HostPinned);
  reduce_result.resize(1);
  {
    //nb_loop = 1;
    //nb_value = 1000000;
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      CooperativeWorkGroupLoopRange loop_range(nb_value);
      auto partial_sum_span = viewInOut(command, by_block_partial_sum);
      auto out_reduce_result = viewOut(command, reduce_result);
      command << RUNCOMMAND_LAUNCH(iter, loop_range)
      {
        auto grid = iter.grid();
        auto block = iter.block();
        auto w = iter.workItem();

        Int64 my_v = 0;
        for (Int32 i : w.linearIndexes())
          my_v += c_view[i];
        block.barrier();
        Int32 nb_block = 0;
#if defined(ARCCORE_COMPILING_CUDA_OR_HIP) && defined(ARCCORE_DEVICE_CODE)
        nb_block = grid.nbBlock();
        Int64 v = Arcane::Accelerator::Impl::block_reduce<Arcane::Accelerator::Impl::ReduceFunctorSum<Int64>, 32, Int64>(my_v);
        if (w.rankInBlock() == 0) {
          //printf("V0=%d %ld\n", block.groupRank(), v);
          partial_sum_span[block.groupRank()] = v;
        }
#endif
        grid.barrier();
        if (w.rankInBlock() == 0 && block.groupRank() == 0) {
          Int64 final_sum = 0;
          for (Int32 i = 0; i < nb_block; ++i) {
            Int64 v = partial_sum_span[i];
            //printf("ADD_V block=%d v=%ld\n", i, v);
            final_sum += partial_sum_span[i];
          }
          partial_sum_span[0] = final_sum;
          out_reduce_result[0] = final_sum;
#if !defined(__INTEL_LLVM_COMPILER)
          // oneDPC++ ne possède pas de printf.
          //printf("FINAL= nb_block=%d v=%ld\n", nb_block, final_sum);
#endif
        }
      };
      total_x += reduce_result[0];
    }
  }
  double y = Platform::getRealTime();
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalCooperativeLaunch=" << total_x
            << " nb_part=" << nb_part << " nb_value=" << nb_value
            << " nb_thread=" << nb_thread
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
  return total_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
