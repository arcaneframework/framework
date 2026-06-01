// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Math.h                                                      (C) 2000-2024 */
/*                                                                           */
/* Diverse mathematical functions.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATH_H
#define ARCANE_UTILS_MATH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Convert.h"

#include <cmath>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \brief Namespace for mathematical functions.
 
  This namespace contains all mathematical functions used
  by the code.
*/
namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Natural logarithm of \a v.
 */
ARCCORE_HOST_DEVICE inline double
log(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v == 0.0 || v < 0.0)
    arcaneMathError(v, "log");
#endif
  return std::log(v);
}

/*!
 * \brief Natural logarithm of \a v.
 */
ARCCORE_HOST_DEVICE inline long double
log(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v == 0.0 || v < 0.0)
    arcaneMathError(v, "log");
#endif
  return std::log(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Decimal logarithm of \a v.
 */
ARCCORE_HOST_DEVICE inline double
log10(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v == 0.0 || v < 0.0)
    arcaneMathError(v, "log");
#endif
  return std::log10(v);
}

/*!
 * \brief Decimal logarithm of \a v.
 */
ARCCORE_HOST_DEVICE inline long double
log10(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v == 0.0 || v < 0.0)
    arcaneMathError(v, "log");
#endif
  return std::log10(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Round \a v down to the immediately lower integer.
 */
ARCCORE_HOST_DEVICE inline double
floor(double v)
{
  return std::floor(v);
}

/*!
 * \brief Round \a v down to the immediately lower integer.
 */
ARCCORE_HOST_DEVICE inline long double
floor(long double v)
{
  return std::floor(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Exponential of \a v.
 */
ARCCORE_HOST_DEVICE inline double
exp(double v)
{
  return std::exp(v);
}

/*!
 * \brief Exponential of \a v.
 */
ARCCORE_HOST_DEVICE inline long double
exp(long double v)
{
  return std::exp(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Square root of \a v.
 */
ARCCORE_HOST_DEVICE inline double
sqrt(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v < 0.)
    arcaneMathError(v, "sqrt");
#endif
  return std::sqrt(v);
}

/*!
 * \brief Square root of \a v.
 */
ARCCORE_HOST_DEVICE inline long double
sqrt(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v < 0.)
    arcaneMathError(v, "sqrt");
#endif
  return std::sqrt(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Power function.
 *
 * Calculates \a x to the power of \a y.
 *
 * \pre x>=0 or y is an integer
 */
ARCCORE_HOST_DEVICE inline double
pow(double x, double y)
{
#ifdef ARCANE_CHECK_MATH
  // Invalid arguments if x is negative and y is not an integer
  if (x < 0.0 && ::floor(y) != y)
    arcaneMathError(x, y, "pow");
#endif
  return std::pow(x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Power function.
 *
 * Calculates \a x to the power of \a y.
 *
 * \pre x>=0 or y is an integer
 */
ARCCORE_HOST_DEVICE inline long double
pow(long double x, long double y)
{
#ifdef ARCANE_CHECK_MATH
  // Invalid arguments if x is negative and y is not an integer
  if (x < 0.0 && ::floorl(y) != y)
    arcaneMathError(x, y, "pow");
#endif
  return std::pow(x, y);
}

/*!
 * \brief Power function.
 *
 * Calculates \a x to the power of \a y.
 *
 * \pre x>=0 or y is an integer
 */
ARCCORE_HOST_DEVICE inline long double
pow(double x, long double y)
{
#ifdef ARCANE_CHECK_MATH
  // Invalid arguments if x is negative and y is not an integer
  if (x < 0.0 && ::floorl(y) != y)
    arcaneMathError(x, y, "pow");
#endif
  return std::pow(x, y);
}

/*!
 * \brief Power function.
 *
 * Calculates \a x to the power of \a y.
 *
 * \pre x>=0 or y is an integer
 */
ARCCORE_HOST_DEVICE inline long double
pow(long double x, double y)
{
#ifdef ARCANE_CHECK_MATH
  // Invalid arguments if x is negative and y is not an integer
  if (x < 0.0 && ::floor(y) != y)
    arcaneMathError(x, y, "pow");
#endif
  return std::pow(x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the minimum of two elements.
 *
 * \ingroup GroupMathUtils
 *
 * Uses the < operator to determine the minimum.
 */
template <class T> ARCCORE_HOST_DEVICE inline T
min(const T& a, const T& b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(long double a, long double b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(double a, long double b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(long double a, double b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
min(double a, double b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
min(float a, float b)
{
  return ((a < b) ? a : b);
}

/*!
 * \brief Returns the minimum of two integers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline int
min(int a, int b)
{
  return ((a < b) ? a : b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the maximum of two elements.
 *
 * \ingroup GroupMathUtils
 *
 * Uses the < operator to determine the maximum.
 */
template <class T> ARCCORE_HOST_DEVICE inline T
max(const T& a, const T& b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(long double a, long double b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(double a, long double b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(long double a, double b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two integers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline unsigned long
max(unsigned long a, unsigned long b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
max(double a, double b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two real numbers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
max(float a, float b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two Int16
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int16
max(Int16 a, Int16 b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two Int32
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int32
max(Int32 a, Int32 b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two Int32
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int32 a, Int64 b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two Int64
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int64 a, Int32 b)
{
  return ((a < b) ? b : a);
}

/*!
 * \brief Returns the maximum of two Int64
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int64 a, Int64 b)
{
  return ((a < b) ? b : a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the absolute value of a real number.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
abs(long double a)
{
  return std::abs(a);
}

/*!
 * \brief Returns the absolute value of a real number.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
abs(double a)
{
  return std::abs(a);
}

/*!
 * \brief Returns the absolute value of a real number.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
abs(float a)
{
  return std::abs(a);
}

/*!
 * \brief Returns the absolute value of an 'int'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline short
abs(short a)
{
  return (a > 0) ? a : (short)(-a);
}

/*!
 * \brief Returns the absolute value of an 'int'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline int
abs(int a)
{
  return (a > 0) ? a : (-a);
}

/*!
 * \brief Returns the absolute value of a 'long'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long
abs(long a)
{
  return (a > 0L) ? a : (-a);
}

/*!
 * \brief Returns the absolute value of a 'long'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long long
abs(long long a)
{
  return (a > 0LL) ? a : (-a);
}

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
