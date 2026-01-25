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
namespace
{
  void _setGroupSize(Accelerator::RunCommand& command, Int32 group_size)
  {
    if ((group_size % 32) != 0)
      ARCCORE_FATAL("group_size '{0}' is not a multiple of 32", group_size);
    // TODO: vérifier que la valeur ne sera pas surchargée par la suite.
    command.addNbThreadPerBlock(group_size);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Accelerator::WorkGroupLoopRange<Int32> Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size)
{
  Int32 total_size = nb_group * group_size;
  _setGroupSize(command, group_size);
  return { total_size, nb_group, group_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_element
Accelerator::WorkGroupLoopRange<Int32> Accelerator::
makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size)
{
  // Calcule automatiquement la taille d'un groupe si l'argument vaut '0'.
  if (group_size == 0) {
    if (nb_group != 0)
      ARCCORE_FATAL("Value of argument 'nb_group' has to be '0' if 'group_size' is '0'");
    // TODO: pour l'instant on met 256 par défaut mais il faudrait peut-être
    // mettre une valeur plus grande sur CPU.
    group_size = 256;
    nb_group = nb_element / group_size;
  }
  if ((nb_element % group_size) != 0)
    ++nb_group;

  _setGroupSize(command, group_size);
  return { nb_element, nb_group, group_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Accelerator::CooperativeWorkGroupLoopRange<Int32> Accelerator::
makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size)
{
  Int32 total_size = nb_group * group_size;
  _setGroupSize(command, group_size);
  return { total_size, nb_group, group_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un intervalle d'itération pour la commande \a command pour \a nb_element
Accelerator::CooperativeWorkGroupLoopRange<Int32> Accelerator::
makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size)
{
  // Calcule automatiquement la taille d'un groupe si l'argument vaut '0'.
  if (group_size == 0) {
    if (nb_group != 0)
      ARCCORE_FATAL("Value of argument 'nb_group' has to be '0' if 'group_size' is '0'");
    // TODO: pour l'instant on met 256 par défaut mais il faudrait peut-être
    // mettre une valeur plus grande sur CPU.
    group_size = 256;
    nb_group = nb_element / group_size;
  }
  if ((nb_element % group_size) != 0)
    ++nb_group;

  _setGroupSize(command, group_size);
  return { nb_element, nb_group, group_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_> HostLaunchLoopRangeBase<IndexType_>::
HostLaunchLoopRangeBase(IndexType total_size, Int32 nb_group, Int32 block_size)
: m_total_size(total_size)
, m_nb_group(nb_group)
, m_group_size(block_size)
{
  m_last_group_size = (total_size - (block_size * (nb_group - 1)));
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
