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
#ifndef ARCANE_UTILS_NUMERICTRAITS_H
#define ARCANE_UTILS_NUMERICTRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NumericTraits.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
