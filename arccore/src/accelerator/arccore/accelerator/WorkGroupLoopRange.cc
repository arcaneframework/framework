// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/WorkGroupLoopRange.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ParallelLoopOptions.h"

#include "arccore/common/accelerator/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <bool IsCooperativeLaunch, typename IndexType_> ARCCORE_ACCELERATOR_EXPORT void
WorkGroupLoopRangeBase<IsCooperativeLaunch, IndexType_>::
setBlockSize(IndexType block_size)
{
  if ((block_size <= 0) || ((block_size % 32) != 0))
    ARCCORE_FATAL("Invalid value '{0}' for block size: should be a multiple of 32", block_size);
  m_block_size = block_size;
  _setNbBlock();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <bool IsCooperativeLaunch, typename IndexType_> ARCCORE_ACCELERATOR_EXPORT void
WorkGroupLoopRangeBase<IsCooperativeLaunch, IndexType_>::
setBlockSize(RunCommand& command)
{
  // TODO: en multi-threading, à calculer en fonction du nombre de threads
  // disponibles et du nombre total d'éléments
  IndexType block_size = 1024;
  eExecutionPolicy policy = command.executionPolicy();
  if (isAcceleratorPolicy(policy))
    block_size = 256;
  else if (policy == eExecutionPolicy::Thread) {
    ParallelLoopOptions loop_options = command.parallelLoopOptions();
    Int32 nb_thread = loop_options.maxThread();
    if (nb_thread == 0)
      nb_thread = 1;
    m_nb_block = nb_thread;
    m_block_size = (m_nb_element + (nb_thread - 1)) / nb_thread;
    loop_options.setGrainSize(1);
    command.setParallelLoopOptions(loop_options);
    return;
  }
  else if (IsCooperativeLaunch) {
    // TODO: gérer le multi-threading.
    // En séquentiel, il n'y a qu'un seul bloc dont la taille est le nombre
    // d'éléments.
    m_block_size = m_nb_element;
    m_nb_block = 1;
    return;
  }
  setBlockSize(block_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <bool IsCooperativeLaunch, typename IndexType_> void
WorkGroupLoopRangeBase<IsCooperativeLaunch, IndexType_>::
_setNbBlock()
{
  m_nb_block = static_cast<Int32>((m_nb_element + (m_block_size - 1)) / m_block_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class WorkGroupLoopRangeBase<true, Int32>;
template class WorkGroupLoopRangeBase<true, Int64>;
template class WorkGroupLoopRangeBase<false, Int32>;
template class WorkGroupLoopRangeBase<false, Int64>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
