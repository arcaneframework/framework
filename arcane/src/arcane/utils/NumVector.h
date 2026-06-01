// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumVector.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Fixed-size vector of numerical types.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMVECTOR_H
#define ARCANE_UTILS_NUMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Small fixed-size vector of N numerical data points.
 *
 * \note Currently only implemented for the Real type.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 *
 * It is possible to access each component of the vector using 'operator[]'
 * or 'operator()' or via the methods vx(), vy(), vz() if the dimension is
 * sufficient (for example, vz() is only accessible if Size>=3.
 */
template <typename T, int Size>
class NumVector
{
  static_assert(Size > 1, "Size has to be strictly greater than 1");
  static_assert(std::is_same_v<T,Real>,"Only type 'Real' is allowed");

 public:

  using ThatClass = NumVector<T, Size>;
  using DataType = T;

 public:

  //! Constructs the zero vector.
  NumVector() = default;

  //! Constructs with the pair (ax,ay)
  constexpr ARCCORE_HOST_DEVICE NumVector(T ax, T ay) requires(Size == 2)

  {
    m_values[0] = ax;
    m_values[1] = ay;
  }

  //! Constructs with the triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE NumVector(T ax, T ay, T az) requires(Size == 3)

  {
    m_values[0] = ax;
    m_values[1] = ay;
    m_values[2] = az;
  }

  //! Constructs with the quadruplet (a1,a2,a3,a4)
  constexpr ARCCORE_HOST_DEVICE NumVector(T a1, T a2, T a3, T a4) requires(Size == 4)

  {
    m_values[0] = a1;
    m_values[1] = a2;
    m_values[2] = a3;
    m_values[3] = a4;
  }

  //! Constructs with the quintuplet (a1,a2,a3,a4,a5)
  constexpr ARCCORE_HOST_DEVICE NumVector(T a1, T a2, T a3, T a4, T a5) requires(Size == 5)
  {
    m_values[0] = a1;
    m_values[1] = a2;
    m_values[2] = a3;
    m_values[3] = a4;
    m_values[4] = a5;
  }

  //! Constructs the instance with the value \a v for each component
  template <bool = true>
  explicit constexpr ARCCORE_HOST_DEVICE NumVector(const T (&v)[Size])
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v[i];
  }

  //! Constructs the instance with the value \a v for each component
  explicit constexpr ARCCORE_HOST_DEVICE NumVector(std::array<T, Size> v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v[i];
  }

  //! Constructs the instance with the value \a v for each component
  explicit constexpr ARCCORE_HOST_DEVICE NumVector(T v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
  }

  explicit constexpr ARCCORE_HOST_DEVICE NumVector(Real2 v) requires(Size == 2)
  : NumVector(v.x, v.y)
  {}

  explicit constexpr ARCCORE_HOST_DEVICE NumVector(Real3 v) requires(Size == 3)
  : NumVector(v.x, v.y, v.z)
  {}

  //! Assigns the triplet (v,v,v) to the instance.
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(Real v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real2& v) requires(Size == 2)
  {
    *this = ThatClass(v);
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real3& v) requires(Size == 3)
  {
    *this = ThatClass(v);
    return (*this);
  }

  operator Real2() const requires(Size == 2) { return Real2(m_values[0], m_values[1]); }

  operator Real3() const requires(Size == 3) { return Real3(m_values[0], m_values[1], m_values[2]); }

 public:

  constexpr ARCCORE_HOST_DEVICE static ThatClass zero() { return ThatClass(); }

 public:

  constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const
  {
    bool is_nearly_zero = true;
    for (int i = 0; i < Size; ++i)
      is_nearly_zero = is_nearly_zero && math::isNearlyZero(m_values[i]);
    return is_nearly_zero;
  }

  //! Returns the square of the L2 norm of the triplet \f$x^2+y^2+z^2\f$
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const
  {
    T v = T();
    for (int i = 0; i < Size; ++i)
      v += m_values[i] * m_values[i];
    return v;
  }

  //! Returns the L2 norm of the triplet \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_HOST_DEVICE Real normL2() const { return _sqrt(squareNormL2()); }

  //! Absolute value component by component.
  ARCCORE_HOST_DEVICE ThatClass absolute() const
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = math::abs(m_values[i]);
    return v;
  }

  //! Adds \a b to each component of the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator+=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] += b;
    return (*this);
  }

  //! Adds \a b to the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator+=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] += b.m_values[i];
    return (*this);
  }

  //! Subtracts \a b from each component of the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator-=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] -= b;
    return (*this);
  }

  //! Subtracts \a b from the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator-=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] -= b.m_values[i];
    return (*this);
  }

  //! Multiplies each component by \a b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator*=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] *= b;
    return (*this);
  }

  //! Divides each component by \a b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator/=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] /= b;
    return (*this);
  }

  //! Creates a triplet that equals this triplet added to \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] + b.m_values[i];
    return v;
  }

  //! Creates a triplet that equals \a b subtracted from this triplet
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] - b.m_values[i];
    return v;
  }

  //! Creates a triplet opposite to the current triplet
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = -m_values[i];
    return v;
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(T a, const ThatClass& vec)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a * vec.m_values[i];
    return v;
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& vec, T b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = vec.m_values[i] * b;
    return v;
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator/(const ThatClass& vec, T b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = vec.m_values[i] / b;
    return v;
  }

  /*!
   * \brief Compares the current instance component by component to \a b.
   *
   * \retval true if this.x==b.x and this.y==b.y and this.z==b.z.
   * \retval false otherwise.
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator==(const ThatClass& a, const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      if (!_eq(a.m_values[i], b.m_values[i]))
        return false;
    return true;
  }

  /*!
   * \brief Compares two vectors
   * For the notion of equality, see operator==()
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return !(a == b);
  }

  constexpr ARCCORE_HOST_DEVICE T& operator()(Int32 i)
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }
  constexpr ARCCORE_HOST_DEVICE T operator()(Int32 i) const
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }
  constexpr ARCCORE_HOST_DEVICE T& operator[](Int32 i)
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }
  constexpr ARCCORE_HOST_DEVICE T operator[](Int32 i) const
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }

  //! Value of the first component
  T& vx() requires(Size >= 1)
  {
    return m_values[0];
  }
  //! Value of the first component
  T vx() const requires(Size >= 1)
  {
    return m_values[0];
  }

  //! Value of the second component
  T& vy() requires(Size >= 2)
  {
    return m_values[1];
  }
  //! Value of the second component
  T vy() const requires(Size >= 2)
  {
    return m_values[1];
  }

  //! Value of the third component
  T& vz() requires(Size >= 3)
  {
    return m_values[2];
  }
  //! Value of the third component
  T vz() const requires(Size >= 3)
  {
    return m_values[2];
  }

 private:

  //! Vector values
  T m_values[Size] = {};

 private:

  /*!
   * \brief Compares the values of \a a and \a b using the TypeEqualT comparator
   * \retval true if \a a and \a b are equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool
  _eq(T a, T b)
  {
    return math::isEqual(a, b);
  }

  //! Returns the square root of \a a
  ARCCORE_HOST_DEVICE static T _sqrt(T a)
  {
    return math::sqrt(a);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
