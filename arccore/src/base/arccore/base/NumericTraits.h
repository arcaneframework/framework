// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumericTraits.h                                             (C) 2000-2025 */
/*                                                                           */
/* Vue sur un tableaux multi-dimensionnel pour les types numériques.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_NUMERICTRAITS_H
#define ARCCORE_BASE_NUMERICTRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour les types numériques.
 *
 * Permet d'indiquer si un type numérique à un ou plusieurs opérateurs
 * 'operator[]' ainsi que leur type de retour.
 */
template<typename DataType>
class NumericTraitsT
{
  //! Type de retour de operator[]
  // using SubscriptType = Real2;

  //! Type de retour de operator[] const
  // using SubscriptConstType = Real2;

  //! Type de retour de operator[][]
  // using Subscript2Type = Real;

  //! Type de retour de operator[][] const
  // using Subscript2ConstType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
