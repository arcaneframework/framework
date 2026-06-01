// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathApfloat.h                                               (C) 2000-2020 */
/*                                                                           */
/* Various mathematical functions for the apfloat type.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATHAPFLOAT_H
#define ARCANE_UTILS_MATHAPFLOAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <apfloat.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Natural logarithm of \a v.
 */
inline apfloat
log(apfloat v)
{
#ifdef ARCANE_CHECK_MATH
  if (v == 0.0 || v < 0.0)
    arcaneMathError(Convert::toDouble(v), "log");
#endif
  return ::log(v);
}

/*!
 * \brief Round \a v down to the immediately lower integer.
 */
inline apfloat
floor(apfloat v)
{
  return ::floor(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Exponential of \a v.
 */
inline apfloat
exp(apfloat v)
{
  return ::exp(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Square root of \a v.
 */
inline apfloat
sqrt(apfloat v)
{
#ifdef ARCANE_CHECK_MATH
  if (v < 0.)
    arcaneMathError(Convert::toDouble(v), "sqrt");
#endif
  return ::sqrt(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Power function.
 *
 * Calculates \a x raised to the power of \a y.
 *
 * \pre x>=0 or y is an integer
 */
inline apfloat
pow(apfloat x, apfloat y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x < 0.0 && ::floor(y) != y)
    arcaneMathError(Convert::toDouble(x), Convert::toDouble(y), "pow");
#endif
  return ::pow(x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
inline apfloat
min(apfloat a, apfloat b)
{
  return ((a < b) ? a : b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
inline apfloat
max(apfloat a, apfloat b)
{
  return ((a < b) ? b : a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the absolute value of a real number.
 * \ingroup GroupMathUtils
 */
inline apfloat
abs(apfloat a)
{
  return ::abs(a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
