// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumericTraits.h                                             (C) 2000-2025 */
/*                                                                           */
/* View of a multi-dimensional array for numeric types.                      */
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
 * \brief Information for numeric types.
 *
 * Allows indicating if a numeric type has one or more operators
 * 'operator[]' as well as their return type.
 */
template<typename DataType>
class NumericTraitsT
{
  //! Return type of operator[]
  // using SubscriptType = Real2;

  //! Return type of operator[] const
  // using SubscriptConstType = Real2;

  //! Return type of operator[][]
  // using Subscript2Type = Real;

  //! Return type of operator[][] const
  // using Subscript2ConstType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
