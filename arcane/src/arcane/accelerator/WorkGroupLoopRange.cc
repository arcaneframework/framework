// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/WorkGroupLoopRange.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

WorkGroupLoopRange::
WorkGroupLoopRange(Int32 total_size, Int32 nb_group, Int32 block_size)
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

namespace Arcane
{

Accelerator::WorkGroupLoopRange Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size)
{
  Int32 total_size = nb_group * group_size;
  // TODO: vérifier que la valeur ne sera pas surchargée par la suite.
  command.addNbThreadPerBlock(group_size);
  return WorkGroupLoopRange(total_size, nb_group, group_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_element
Accelerator::WorkGroupLoopRange Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element)
{
  const Int32 group_size = 256;
  Int32 nb_group = nb_element / group_size;
  if ((nb_element % group_size) != 0)
    ++nb_group;
  // TODO: vérifier que la valeur ne sera pas surchargée par la suite.
  command.addNbThreadPerBlock(group_size);
  return WorkGroupLoopRange(nb_element, nb_group, group_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
