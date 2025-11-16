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
/*!
 * \file RunCommandLaunch.h
 *
 * \brief Types et macros pour gérer le parallélisme hiérarchique sur les accélérateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
  void _setGroupSize(Accelerator::RunCommand& command, Int32 group_size)
  {
    if ((group_size % 32) != 0)
      ARCANE_FATAL("group_size '{0}' is not a multiple of 32", group_size);
    // TODO: vérifier que la valeur ne sera pas surchargée par la suite.
    command.addNbThreadPerBlock(group_size);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Accelerator::WorkGroupLoopRange Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size)
{
  Int32 total_size = nb_group * group_size;
  _setGroupSize(command, group_size);
  return WorkGroupLoopRange(total_size, nb_group, group_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_element
Accelerator::WorkGroupLoopRange Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size)
{
  // Calcule automatiquement la taille d'un groupe si l'argument vaut '0'.
  if (group_size == 0) {
    if (nb_group != 0)
      ARCANE_FATAL("Value of argument 'nb_group' has to be '0' if 'group_size' is '0'");
    // TODO: pour l'instant on met 256 par défaut mais il faudrait peut-être
    // mettre une valeur plus grande sur CPU.
    group_size = 256;
    nb_group = nb_element / group_size;
  }
  if ((nb_element % group_size) != 0)
    ++nb_group;

  _setGroupSize(command, group_size);
  return WorkGroupLoopRange(nb_element, nb_group, group_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
