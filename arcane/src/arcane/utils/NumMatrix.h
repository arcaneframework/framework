// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumMatrix.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Fixed-size square matrix of numeric types.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMMATRIX_H
#define ARCANE_UTILS_NUMMATRIX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumVector.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Small fixed-size matrix containing RowSize rows and ColumnSize columns.
 *
 * \note Currently only implemented for the Real type.
 *
 * \warning API is under definition. Do not use outside of Arcane
 *
 * It is possible to access each vector component using 'operator[]'
 * or 'operator()' or via the methods vx(), vy(), vz() if the dimension is
 * sufficient (for example, vz() is only accessible if Size>=3.
 */
template <typename T, int RowSize, int ColumnSize>
class NumMatrix
{
  static_assert(RowSize > 1, "RowSize has to be strictly greater than 1");
  static_assert(ColumnSize > 1, "RowSize has to be strictly greater than 1");
  //static_assert(RowSize == ColumnSize, "Only square matrix are allowed (ColumnSize==RowSize)");
  static_assert(std::is_same_v<T, Real>, "Only type 'Real' is allowed");
  static constexpr int Size = RowSize;
  static constexpr bool isSquare() { return RowSize == ColumnSize; }
  static constexpr bool isSquare2() { return RowSize == 2 && ColumnSize == 2; }
  static constexpr bool isSquare3() { return RowSize == 3 && ColumnSize == 3; }

 public:

  using VectorType = NumVector<T, ColumnSize>;
  using ThatClass = NumMatrix<T, RowSize, ColumnSize>;
  using DataType = T;

 public:

  //! Constructs the matrix with all coefficients zero.
  NumMatrix() = default;

  //! Constructs the matrix with rows (ax, ay)
  constexpr ARCCORE_HOST_DEVICE NumMatrix(const VectorType& ax, const VectorType& ay)
  requires(RowSize == 2)
  {
    m_values[0] = ax;
    m_values[1] = ay;
  }

  //! Constructs the matrix with rows (ax, ay, az)
  constexpr ARCCORE_HOST_DEVICE NumMatrix(const VectorType& ax, const VectorType& ay, const VectorType& az)
  requires(RowSize == 3)
  {
    m_values[0] = ax;
    m_values[1] = ay;
    m_values[2] = az;
  }

  //! Constructs the matrix with rows (a1, a2, a3, a4)
  constexpr ARCCORE_HOST_DEVICE NumMatrix(const VectorType& a1, const VectorType& a2,
                                          const VectorType& a3, const VectorType& a4)
  requires(RowSize == 4)
  {
    m_values[0] = a1;
    m_values[1] = a2;
    m_values[2] = a3;
    m_values[3] = a4;
  }

  //! Constructs the matrix with rows (a1, a2, a3, a4, a5)
  constexpr ARCCORE_HOST_DEVICE NumMatrix(const VectorType& a1, const VectorType& a2,
                                          const VectorType& a3, const VectorType& a4,
                                          const VectorType& a5)
  requires(RowSize == 5)
  {
    m_values[0] = a1;
    m_values[1] = a2;
    m_values[2] = a3;
    m_values[3] = a4;
    m_values[4] = a5;
  }

  //! Constructs the instance with the triplet (v, v, v).
  constexpr ARCCORE_HOST_DEVICE explicit NumMatrix(T v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
  }

  explicit constexpr ARCCORE_HOST_DEVICE NumMatrix(Real2x2 v) requires(isSquare2())
  : NumMatrix(VectorType(v.x), VectorType(v.y))
  {}

  explicit constexpr ARCCORE_HOST_DEVICE NumMatrix(Real3x3 v) requires(isSquare3())
  : NumMatrix(VectorType(v.x), VectorType(v.y), VectorType(v.z))
  {}

  //! Assigns the triplet (v, v, v) to the instance.
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(T v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real2x2& v) requires(isSquare2())
  {
    *this = ThatClass(v);
    return (*this);
  }

  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real3x3& v) requires(isSquare3())
  {
    *this = ThatClass(v);
    return (*this);
  }

  operator Real2x2() const requires(isSquare2())
  {
    return Real2x2(m_values[0], m_values[1]);
  }

  operator Real3x3() const requires(isSquare3())
  {
    return Real3x3(m_values[0], m_values[1], m_values[2]);
  }

 public:

  //! Constructs the zero matrix
  constexpr ARCCORE_HOST_DEVICE static ThatClass zero()
  {
    return ThatClass();
  }

  //! Constructs the matrix ((ax,bx,cx), (ay,by,cy), (az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static ThatClass fromColumns(T ax, T ay, T az, T bx, T by, T bz, T cx, T cy, T cz)
  requires(isSquare3())
  {
    return ThatClass(VectorType(ax, bx, cx), VectorType(ay, by, cy), VectorType(az, bz, cz));
  }

  //! Constructs the matrix ((ax,bx,cx), (ay,by,cy), (az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static ThatClass fromLines(T ax, T bx, T cx, T ay, T by, T cy, T az, T bz, T cz)
  requires(isSquare3())
  {
    return ThatClass(VectorType(ax, bx, cx), VectorType(ay, by, cy), VectorType(az, bz, cz));
  }

 public:

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
  constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const
  {
    bool is_nearly_zero = true;
    for (int i = 0; i < Size; ++i)
      is_nearly_zero = is_nearly_zero && math::isNearlyZero(m_values[i]);
    return is_nearly_zero;
  }

  //! Adds b to the triplet.
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator+=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] += b.m_values[i];
    return (*this);
  }
  //! Subtracts b from the triplet
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator-=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] -= b.m_values[i];
    return (*this);
  }
  //! Multiplies each component of the matrix by the real number b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator*=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] *= b;
    return (*this);
  }
  //! Divides each component of the matrix by the real number b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator/=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] *= b;
    return (*this);
  }
  //! Creates a triplet that equals this triplet added to b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] + b.m_values[i];
    return v;
  }
  //! Creates a triplet that equals a subtracted from this triplet
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] - b.m_values[i];
    return v;
  }
  //! Creates a tensor opposite to the current tensor
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = -m_values[i];
    return v;
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(DataType a, const ThatClass& mat)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a * mat.m_values[i];
    return v;
  }
  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& mat, DataType b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = mat.m_values[i] * b;
    return v;
  }
  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator/(const ThatClass& mat, DataType b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = mat.m_values[i] / b;
    return v;
  }

  /*!
   * \brief Compares the current instance component by component to b.
   *
   * \retval true if this.x==b.x and this.y==b.y and this.z==b.z.
   * \retval false otherwise.
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator==(const ThatClass& a, const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      if (a.m_values[i] != b.m_values[i])
        return false;
    return true;
  }

  /*!
   * \brief Compares two triplets.
   * For the notion of equality, see operator==()
   * \retval true if the two triplets are different,
   * \retval false otherwise.
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return !(a == b);
  }

 public:

  // Retrieves the i-th row
  constexpr ARCCORE_HOST_DEVICE VectorType operator()(Int32 i) const
  {
    ARCCORE_CHECK_AT(i, RowSize);
    return m_values[i];
  }

  // Retrieves the i-th row
  constexpr ARCCORE_HOST_DEVICE VectorType operator[](Int32 i) const
  {
    ARCCORE_CHECK_AT(i, RowSize);
    return m_values[i];
  }

  // Retrieves a reference to the value of the i-th row and j-th column
  constexpr ARCCORE_HOST_DEVICE T& operator()(Int32 i, Int32 j)
  {
    ARCCORE_CHECK_AT(i, RowSize);
    ARCCORE_CHECK_AT(j, ColumnSize);
    return m_values[i](j);
  }

  // Retrieves the value of the i-th row and j-th column
  constexpr ARCCORE_HOST_DEVICE T operator()(Int32 i, Int32 j) const
  {
    ARCCORE_CHECK_AT(i, RowSize);
    ARCCORE_CHECK_AT(j, ColumnSize);
    return m_values[i](j);
  }

  //! Sets the value of the i-th row to v
  constexpr ARCCORE_HOST_DEVICE void setLine(Int32 i, const VectorType& v)
  {
    ARCCORE_CHECK_AT(i, RowSize);
    m_values[i] = v;
  }

 public:

  VectorType& vx() requires(RowSize >= 1)
  {
    return m_values[0];
  }

  VectorType vx() const requires(RowSize >= 1)
  {
    return m_values[0];
  }

  VectorType& vy() requires(RowSize >= 2)
  {
    return m_values[1];
  }

  VectorType vy() const requires(RowSize >= 2)
  {
    return m_values[1];
  }

  VectorType& vz() requires(RowSize >= 3)
  {
    return m_values[2];
  }

  VectorType vz() const requires(RowSize >= 3)
  {
    return m_values[2];
  }

 private:

  VectorType m_values[RowSize] = {};

 private:

  /*!
   * \brief Compares the values of a and b using the TypeEqualT comparator
   * \retval true if a and b are equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool _eq(T a, T b)
  {
    return TypeEqualT<T>::isEqual(a, b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
