// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLaunch.cc                                         (C) 2000-2025 */
/*                                                                           */
/* RunCommand pour le parallélisme hiérarchique.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
