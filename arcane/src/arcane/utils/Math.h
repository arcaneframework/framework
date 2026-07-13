// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Math.h                                                      (C) 2000-2026 */
/*                                                                           */
/* Diverse mathematical functions.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATH_H
#define ARCANE_UTILS_MATH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#ifndef ARCCORE_COMPILING_FRAMEWORK
#include "arcane/utils/Convert.h"
#endif
#include "arccore/base/MathBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Truncates the precision of the real number v to nb_digit significant figures.
 *
 * For a double-precision real number in IEEE 754, the number of significant bits
 * is 15 or 16 depending on the value. It should be noted that it is not possible
 * to simply and quickly truncate the precision to a given value.
 * This is why nb_digit represents an approximate number of digits.
 * Notably, it is not possible to go below 8 significant figures.
 *
 * If nb_digit is less than or equal to zero or greater than 15, the value v is returned.
 */
extern ARCANE_UTILS_EXPORT double
truncateDouble(double v, Integer nb_digit);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Truncates the precision of the array of real numbers values to
 * \a nb_digit significant figures.
 *
 * On exit, each element of values is modified as if after calling
 * truncateDouble(double v,Integer nb_digit).
 */
extern ARCANE_UTILS_EXPORT void
truncateDouble(ArrayView<double> values, Integer nb_digit);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_REAL_USE_APFLOAT
#include "arcane/utils/MathApfloat.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
