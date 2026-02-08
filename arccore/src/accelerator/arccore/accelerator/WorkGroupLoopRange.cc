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
#include "arccore/common/accelerator/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <bool IsCooperativeLaunch, typename IndexType_> ARCCORE_ACCELERATOR_EXPORT void
WorkGroupLoopRangeBase<IsCooperativeLaunch, IndexType_>::
setBlockSize(Int32 block_size)
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
setBlockSize(const RunCommand& command)
{
  // TODO: en multi-threading, à calculer en fonction du nombre de threads
  // disponibles et du nombre total d'éléments
  Int32 block_size = 1024;
  if (isAcceleratorPolicy(command.executionPolicy()))
    block_size = 256;
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
