// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Numeric.h                                                   (C) 2000-2020 */
/*                                                                           */
/* Numeric constant definitions.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_NUMERIC_H
#define ARCANE_DATATYPE_NUMERIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Limits.h"
#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Comparison operations for a numeric type T
 *
 * This class defines a comparison operator for the
 * template parameter type 'T'. There are two types of comparisons:
 * - exact comparisons (isEqual());
 * - approximate comparisons (isNearlyEqual()).
 *
 * Both types of comparisons are identical, except for
 * floating-point types or equivalents. In this case, the exact comparison
 * compares the two values bit by bit, and the approximate comparison
 * considers two numbers equal if their relative difference is
 * less than an epsilon.
 */
template<class T>
class TypeEqualT
{
 public:

  /*!
   * \brief Compares \a a to zero.
   * \retval true if \a a is zero within an epsilon,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZero (const T& a)
  {
    return (a==T());
  }

  /*!
   * \brief Compares \a a to zero.
   * \retval true if \a a is exactly zero,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isZero (const T& a)
  {
    return (a==T());
  }

  /*!
   * \brief Compares \a a to \a b.
   * \retval true if \a a and \b are equal within an epsilon,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqual(const T& a,const T& b)
  {
    return (a==b);
  }

  /*!
   * \brief Compares \a a to \a b.
   * \retval true if \a a and \b are equal within an epsilon,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqualWithEpsilon(const T& a,const T& b,const T&)
  {
    return (a==b);
  }

  /*!
   * \brief Compares \a a to \a b.
   * \retval true if \a a and \b are exactly equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isEqual(const T& a,const T& b)
  {
    return (a==b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Defines the == operator for floats.
 *
 * \note Eventually, it should use the 'numeric_limits' class
 * from the STL when it is implemented.
 */
template<class T>
class FloatEqualT
{
 private:
  constexpr ARCCORE_HOST_DEVICE static T nepsilon() { return FloatInfo<T>::nearlyEpsilon(); }
 public:
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZero(T a)
  {
    return ( (a<0.) ? a>-nepsilon() : a<nepsilon() );
  }
  
  /*!
   * \brief Compares \a a to zero within \a epsilon.
   * 
   * \a epsilon must be positive.
   *
   * \retval true if abs(a)<epilon
   * \retval false otherwise
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZeroWithEpsilon(T a,T epsilon)
  {
    return ( (a<0.) ? a>-epsilon : a<epsilon );
  }
  
  /*! \brief Compares \a with \a b*epsilon.
   * \warning b must be positive. */
  ARCCORE_HOST_DEVICE static bool isNearlyZero(T a,T b)
  {
    return ( (a<0.) ? a>-(b*nepsilon()) : a<(b*nepsilon()) );
  }

  constexpr ARCCORE_HOST_DEVICE static bool isTrueZero(T a) { return (a==FloatInfo<T>::zero()); }
  constexpr ARCCORE_HOST_DEVICE static bool isZero(T a) { return (a==FloatInfo<T>::zero()); }
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqual(T a,T b)
  {
    T s = math::abs(a) + math::abs(b);
    T d = a - b;
    return (d==FloatInfo<T>::zero()) ? true : isNearlyZero(d/s);
  }
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqualWithEpsilon(T a,T b,T epsilon)
  {
    T s = math::abs(a) + math::abs(b);
    T d = a - b;
    return (d==FloatInfo<T>::zero()) ? true : isNearlyZeroWithEpsilon(d/s,epsilon);
  }
  constexpr ARCCORE_HOST_DEVICE static bool isEqual(T a,T b)
  {
    return a==b;
  }
};

/*!
 * \internal
 * \brief Specialization of TypeEqualT for the <tt>float</tt> type.
 */
template<>
class TypeEqualT<float>
: public FloatEqualT<float>
{};

/*!
 * \internal
 * \brief Specialization of TypeEqualT for the <tt>double</tt> type.
 */
template<>
class TypeEqualT<double>
: public FloatEqualT<double>
{};

/*!
 * \internal
 * \brief Specialization of TypeEqualT for the <tt>long double</tt> type.
 */
template<>
class TypeEqualT<long double>
: public FloatEqualT<long double>
{};

#ifdef ARCANE_REAL_NOT_BUILTIN
/*!
 * \internal
 * \brief Specialization of TypeEqualT for the <tt>Real</tt> type.
 */
template<>
class TypeEqualT<Real>
: public FloatEqualT<Real>
{};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{

/*!
 * \brief Tests if two values are approximately equal.
 * For integer types, this function is equivalent to IsEqual().
 * In the case of real types, the two numbers are considered equal
 * if and only if the absolute value of their relative difference is
 * less than a given epsilon. This
 * epsilon is equal to float_info<_Type>::nearlyEpsilon().
 * \retval true if the two values are equal,
 * \retval false otherwise.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqual(const _Type& a,const _Type& b)
{
  return TypeEqualT<_Type>::isNearlyEqual(a,b);
}

//! Overload for reals
constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqual(Real a,Real b)
{
  return TypeEqualT<Real>::isNearlyEqual(a,b);
}

/*!
 * \brief Tests if two values are approximately equal.
 * For integer types, this function is equivalent to IsEqual().
 * In the case of real types, the two numbers are considered equal
 * if and only if the absolute value of their relative difference is
 * less than \a epsilon.
 *
 * \retval true if the two values are equal,
 * \retval false otherwise.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqualWithEpsilon(const _Type& a,const _Type& b,const _Type& epsilon)
{
  return TypeEqualT<_Type>::isNearlyEqualWithEpsilon(a,b,epsilon);
}

//! Overload for reals
ARCCORE_HOST_DEVICE constexpr inline bool
isNearlyEqualWithEpsilon(Real a,Real b,Real epsilon)
{
  return TypeEqualT<Real>::isNearlyEqualWithEpsilon(a,b,epsilon);
}

/*!
 * \brief Tests the bit-by-bit equality between two values.
 * \retval true if the two values are equal,
 * \retval false otherwise.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isEqual(const _Type& a,const _Type& b)
{
  return TypeEqualT<_Type>::isEqual(a,b);
}

//! Overload for reals
ARCCORE_HOST_DEVICE constexpr inline bool
isEqual(Real a,Real b)
{
  return TypeEqualT<Real>::isEqual(a,b);
}

/*!
 * \brief Tests if a value is approximately equal to zero within an epsilon.
 *
 * For integer types, this function is equivalent to IsZero().
 * In the case of real types, the value is considered equal to
 * zero if and only if its absolute value is less than an epsilon
 * given by the function float_info<_Type>::nearlyEpsilon().
 * \retval true if the two values are equal,
 * \retval false otherwise.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyZeroWithEpsilon(const _Type& a,const _Type& epsilon)
{
  return TypeEqualT<_Type>::isNearlyZeroWithEpsilon(a,epsilon);
}

/*!
 * \brief Tests if a value is approximately equal to zero using the standard epsilon.
 *
 * The standard epsilon is the one returned by FloatInfo<_Type>::nearlyEpsilon().
 *
 * \sa isNearlyZero(const _Type& a,const _Type& epsilon).
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyZero(const _Type& a)
{
  return TypeEqualT<_Type>::isNearlyZero(a);
}

/*!
 * \brief Tests if a value is exactly equal to zero.
 * \retval true if \a is zero,
 * \retval false otherwise.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isZero(const _Type& a)
{
  return TypeEqualT<_Type>::isZero(a);
}
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
