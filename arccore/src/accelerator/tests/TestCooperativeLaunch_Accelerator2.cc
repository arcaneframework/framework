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
_testCooperativeLaunch2(RunQueue queue, SmallSpan<const Int64> c,
                        Int32 nb_value, Int32 nb_loop)
{
  Int64 total_x = 0;
  // Valeurs partielles par bloc.
  // Doit être dimensionné au nombre maximum de blocs possibles
  NumArray<Int64, MDDim1> by_block_partial_sum(queue.memoryResource());
  by_block_partial_sum.resize(2048);
  // Pour récupèrer le résultat de la réduction.
  NumArray<Int64, MDDim1> reduce_result(eMemoryResource::HostPinned);
  reduce_result.resize(1);
  double x = Platform::getRealTime();
  {
    //nb_loop = 1;
    //nb_value = 100000;
    SmallSpan<const Int64> c_view(c);
    for (int j = 0; j < nb_loop; ++j) {
      auto command = makeCommand(queue);
      CooperativeWorkGroupLoopRange loop_range(nb_value);
      loop_range.setBlockSize(command);
      Int32 local_memory_size = (queue.isAcceleratorPolicy()) ? loop_range.blockSize() : 1;
      LocalMemory<Int64> block_partial_sum(command, local_memory_size);
      auto partial_sum_span = viewInOut(command, by_block_partial_sum);
      auto out_reduce_result = viewOut(command, reduce_result);
      command << RUNCOMMAND_LAUNCH(iter, loop_range, block_partial_sum)
      {
        auto grid = iter.grid();
        auto block = iter.block();
        auto w = iter.workItem();
        auto block_partial_sum_span = block_partial_sum.span();

        // Chaque WorkItem calcule la réduction pour les indices qu'il traite
        // et range le résultat dans la mémoire locale.
        Int64 my_v = 0;
        for (Int32 i : w.linearIndexes())
          my_v += c_view[i];
        block_partial_sum_span[w.rankInBlock()] = my_v;
        // Attend que tous les WorkItem du bloc aient fini
        block.barrier();
        // Le premier WorkItem du bloc fait la réduction
        // sur les valeurs du tableau local.
        if (w.rankInBlock() == 0) {
          Int32 nb_local = block_partial_sum_span.size();
          Int64 block_v = 0;
          for (Int32 i = 0; i < nb_local; ++i)
            block_v += block_partial_sum_span[i];
          partial_sum_span[block.groupRank()] = block_v;
        }
        // Attend que toute la grille ait terminée.
        grid.barrier();
        // Le premier WorkItem fait la réduction finale
        if (w.rankInBlock() == 0 && block.groupRank() == 0) {
          Int64 final_sum = 0;
          Int32 nb_block = grid.nbBlock();
          for (Int32 i = 0; i < nb_block; ++i) {
            final_sum += partial_sum_span[i];
          }
          out_reduce_result[0] = final_sum;
        }
      };
      total_x += reduce_result[0];
    }
  }
  double y = Platform::getRealTime();
  Int64 nb_byte = c.size() * sizeof(Int64) * nb_loop;
  Real diff = y - x;
  Real nb_giga_byte_second = (static_cast<Real>(nb_byte) / 1.0e9) / diff;
  std::cout << "** TotalCooperativeLaunch2=" << total_x
            << " nb_value=" << nb_value
            << " GB/s=" << nb_giga_byte_second << " time=" << diff << "\n";
  return total_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
