// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunch.cc                                         (C) 2000-2026 */
/*                                                                           */
/* RunCommand pour le parallélisme hiérarchique.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file RunCommandLaunch.h
 *
 * \brief Types et macros pour gérer le parallélisme hiérarchique
 * sur les accélérateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_> HostLaunchLoopRangeBase<IndexType_>::
HostLaunchLoopRangeBase(IndexType total_size, Int32 nb_block, Int32 block_size)
: m_total_size(total_size)
, m_nb_block(nb_block)
, m_block_size(block_size)
{
  m_last_block_size = (total_size - (block_size * (nb_block - 1)));
  if (m_last_block_size <= 0)
    ARCCORE_FATAL("Bad value '{0}' for last group size", m_last_block_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ARCCORE_EXPORT HostLaunchLoopRangeBase<Int32>;
template class ARCCORE_EXPORT HostLaunchLoopRangeBase<Int64>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
