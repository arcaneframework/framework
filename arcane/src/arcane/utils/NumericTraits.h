// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumericTraits.h                                             (C) 2000-2022 */
/*                                                                           */
/* Vue sur un tableaux multi-dimensionnel pour les types numériques.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMERICTRAITS_H
#define ARCANE_UTILS_NUMERICTRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

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

template<>
class NumericTraitsT<Real2>
{
 public:
  using SubscriptType = Real&;
  using SubscriptConstType = Real;
};

template<>
class NumericTraitsT<const Real2>
{
 public:
  using SubscriptType = const Real;
  using SubscriptConstType = const Real;
};

template<>
class NumericTraitsT<Real3>
{
 public:
  using SubscriptType = Real&;
  using SubscriptConstType = Real;
};

template<>
class NumericTraitsT<const Real3>
{
 public:
  using SubscriptType = const Real;
  using SubscriptConstType = const Real;
};

template<>
class NumericTraitsT<Real2x2>
{
 public:
  using SubscriptType = const Real2;
  using Subscript2Type = Real&;
  using Subscript2ConstType = const Real;
};

template<>
class NumericTraitsT<const Real2x2>
{
 public:
  using SubscriptType = const Real2;
  using Subscript2Type = const Real;
  using Subscript2ConstType = const Real;
};

template<>
class NumericTraitsT<Real3x3>
{
 public:
  //! Type de retour de operator[] pour ce type
  using SubscriptType = const Real3;
  using Subscript2Type = Real&;
  using Subscript2ConstType = const Real;
};

template<>
class NumericTraitsT<const Real3x3>
{
 public:
  //! Type de retour de operator[] pour ce type
  using SubscriptType = const Real3;
  using Subscript2Type = const Real;
  using Subscript2ConstType = const Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
