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
 * \brief Small fixed-size vector of Size numerical data points.
 *
 * It is possible to access each component of the vector using 'operator[]'
 * or 'operator()' or via the methods vx(), vy(), vz() if the dimension is
 * sufficient (for example, vz() is only accessible if Size>=3.
 */
template <typename T, int Size>
class NumVector
{
  static_assert(Size > 0, "Size has to be strictly greater than 0");

 public:

  using ThatClass = NumVector<T, Size>;
  using DataType = T;
  static constexpr bool isRealType() { return std::is_same_v<T, Real>; }

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

  //! Constructs with the sextuplet (a1,a2,a3,a4,a5,a6)
  constexpr ARCCORE_HOST_DEVICE NumVector(T a1, T a2, T a3, T a4, T a5, T a6) requires(Size == 6)
  {
    m_values[0] = a1;
    m_values[1] = a2;
    m_values[2] = a3;
    m_values[3] = a4;
    m_values[4] = a5;
    m_values[5] = a6;
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

  explicit constexpr ARCCORE_HOST_DEVICE NumVector(Real2 v) requires(Size == 2 && isRealType())
  : NumVector(v.x, v.y)
  {}

  explicit constexpr ARCCORE_HOST_DEVICE NumVector(Real3 v) requires(Size == 3 && isRealType())
  : NumVector(v.x, v.y, v.z)
  {}

  //! Assigns value to all components of the vector
  constexpr ARCCORE_HOST_DEVICE NumVector& operator=(const DataType& value)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = value;
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE NumVector& operator=(const Real2& v)
  requires(Size == 2 && isRealType())
  {
    *this = NumVector(v);
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE NumVector& operator=(const Real3& v)
  requires(Size == 3 && isRealType())
  {
    *this = NumVector(v);
    return (*this);
  }

  constexpr operator Real2() const requires(Size == 2)
  {
    return Real2(m_values[0], m_values[1]);
  }

  constexpr operator Real3() const requires(Size == 3)
  {
    return Real3(m_values[0], m_values[1], m_values[2]);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE static NumVector zero() { return NumVector(); }

 public:

  //! Absolute value component by component.
  ARCCORE_HOST_DEVICE NumVector absolute() const
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = math::abs(m_values[i]);
    return v;
  }

  //! Fill the vector with the value \a v
  constexpr ARCCORE_HOST_DEVICE void fill(const T& v)
  {
    for (int i = 0; i < Size; ++i) {
      m_values[i] = v;
    }
  }

  //! Adds \a b to each component of \a a
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator+=(NumVector& a, T b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] += b;
    return a;
  }

  //! Adds \a b to \a a
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator+=(NumVector& a, const NumVector& b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] += b.m_values[i];
    return a;
  }

  //! Subtracts \a b from each component of \a a
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator-=(NumVector& a, T b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] -= b;
    return a;
  }

  //! Subtracts \a b from each component of \a a
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator-=(NumVector& a, const NumVector& b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] -= b.m_values[i];
    return a;
  }

  //! Multiplies each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator*=(NumVector& a, T b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] *= b;
    return a;
  }

  //! Divides each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE NumVector& operator/=(NumVector& a, T b)
  {
    for (int i = 0; i < Size; ++i)
      a.m_values[i] /= b;
    return a;
  }

  //! Creates a triplet that equals this triplet added to \a b
  friend constexpr ARCCORE_HOST_DEVICE NumVector operator+(const NumVector& a, const NumVector& b)
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] + b.m_values[i];
    return v;
  }

  //! Creates a triplet that equals \a b subtracted from this triplet
  friend constexpr ARCCORE_HOST_DEVICE NumVector operator-(const NumVector& a, const NumVector& b)
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] - b.m_values[i];
    return v;
  }

  //! Creates a triplet opposite to the current triplet
  constexpr ARCCORE_HOST_DEVICE NumVector operator-() const
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = -m_values[i];
    return v;
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE NumVector operator*(T a, const NumVector& vec)
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a * vec.m_values[i];
    return v;
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE NumVector operator*(const NumVector& vec, T b)
  {
    NumVector v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = vec.m_values[i] * b;
    return v;
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE NumVector operator/(const NumVector& vec, T b)
  {
    NumVector v;
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
  friend constexpr ARCCORE_HOST_DEVICE bool operator==(const NumVector& a, const NumVector& b)
  {
    for (int i = 0; i < Size; ++i)
      if (!_eq(a.m_values[i], b.m_values[i]))
        return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& o, const NumVector& t)
  {
    for (int i = 0; i < Size; ++i) {
      if (i != 0)
        o << ' ';
      o << t.m_values[i];
    }
    return o;
  }

  /*!
   * \brief Compares two vectors
   * For the notion of equality, see operator==()
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator!=(const NumVector& a, const NumVector& b)
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
  constexpr ARCCORE_HOST_DEVICE DataType operator[](Int32 i) const
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }

  //! Value of the first component
  constexpr ARCCORE_HOST_DEVICE DataType& vx() requires(Size >= 1)
  {
    return m_values[0];
  }
  //! Value of the first component
  constexpr ARCCORE_HOST_DEVICE DataType vx() const requires(Size >= 1)
  {
    return m_values[0];
  }

  //! Value of the second component
  constexpr ARCCORE_HOST_DEVICE DataType& vy() requires(Size >= 2)
  {
    return m_values[1];
  }
  //! Value of the second component
  constexpr ARCCORE_HOST_DEVICE DataType vy() const requires(Size >= 2)
  {
    return m_values[1];
  }

  //! Value of the third component
  constexpr ARCCORE_HOST_DEVICE DataType& vz() requires(Size >= 3)
  {
    return m_values[2];
  }
  //! Value of the third component
  constexpr ARCCORE_HOST_DEVICE DataType vz() const requires(Size >= 3)
  {
    return m_values[2];
  }

 private:

  //! Vector values
  T m_values[Size] = {};

 private:

  /*!
   * \brief Compares the values of \a a and \a b using the TypeEqualT comparator.
   *
   * \retval true if \a a and \a b are equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool
  _eq(T a, T b)
  {
    return math::isEqual(a, b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compares the vector with the vector zero.
 *
 * The matrix is nearly zero if and only if each of its components
 * is less than a given epsilon. The epsilon value used is that
 * of FloatInfo<DataType>::nearlyEpsilon():
 * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
 */
template <typename DataType, int Size>
constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const NumVector<DataType, Size>& v)
{
  bool is_nearly_zero = true;
  for (int i = 0; i < Size; ++i)
    is_nearly_zero = is_nearly_zero && math::isNearlyZero(v[i]);
  return is_nearly_zero;
}

//! Returns the square of the L2 norm of the triplet \f$x^2+y^2+z^2\f$
template <typename DataType, int Size>
constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const NumVector<DataType, Size>& v)
{
  DataType norm = {};
  for (int i = 0; i < Size; ++i)
    norm += v[i] * v[i];
  return norm;
}

//! Returns the L2 norm of the triplet \f$\sqrt{x^2+y^2+z^2}\f$
template <typename DataType, int Size>
ARCCORE_HOST_DEVICE Real normL2(const NumVector<DataType, Size>& v)
{
  return Arcane::math::sqrt(squareNormL2(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
