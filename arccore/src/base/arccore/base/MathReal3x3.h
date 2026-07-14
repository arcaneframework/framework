// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathReal3x3.h                                               (C) 2000-2026 */
/*                                                                           */
/* 3x3 Matrix of 'Real'.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MATHREAL3X3_H
#define ARCCORE_BASE_MATHREAL3X3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Real3x3.h"
#include "arccore/base/MathReal3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{
/*!
 * \brief Compares the matrix with the zero matrix.
 *
 * The matrix is zero if and only if each of its components
 * is less than a given epsilon. The epsilon value used is that
 * of float_info<value_type>::nearlyEpsilon():
 * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
 *
 * \retval true if the matrix is equal to the zero matrix,
 * \retval false otherwise.
 */
inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real3x3& v)
{
  return isNearlyZero(v.x) && isNearlyZero(v.y) && isNearlyZero(v.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

inline constexpr ARCCORE_HOST_DEVICE bool Real3x3::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
