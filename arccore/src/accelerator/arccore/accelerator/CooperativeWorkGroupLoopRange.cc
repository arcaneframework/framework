// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CooperativeWorkGroupLoopRange.cc                            (C) 2000-2026 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique coopératif.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/CooperativeWorkGroupLoopRange.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CooperativeWorkGroupLoopRange::
CooperativeWorkGroupLoopRange(Int32 total_size, Int32 nb_group, Int32 block_size)
: m_total_size(total_size)
, m_nb_group(nb_group)
, m_group_size(block_size)
{
  m_last_group_size = (total_size - (block_size * (nb_group - 1)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
