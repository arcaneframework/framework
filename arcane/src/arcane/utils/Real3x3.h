// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3x3.h                                                   (C) 2000-2025 */
/*                                                                           */
/* 3x3 Matrix of 'Real'.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3X3_H
#define ARCANE_UTILS_REAL3X3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief POD structure for a Real3x3.
 */
struct Real3x3POD
{
 public:

  Real3POD x;
  Real3POD y;
  Real3POD z;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a 3x3 real matrix.

 The matrix comprises three components \a x, \a y and \a z of type \b Real3.
 Each component represents a row of the matrix.
 Consequently, for a matrix \a m, m.y.z represents the 2nd row
 and the 3rd column of the matrix.

 It is also possible to access the matrix elements like an array. For example
 m[1][2] represents the 2nd row and the 3rd column of the matrix.

 Matrices can be constructed by row by specifying the values
 by row (fromLines()) or by specifying by column (fromColumns()).

 For example:

 * \code
 * Real3x3 matrix;
 * matrix.x.y = 2.0;
 * matrix.y.z = 3.0;
 * matrix.x.x = 5.0;
 * \endcode
*/
class ARCANE_UTILS_EXPORT Real3x3
{
 public:

  //! Constructs the matrix with all coefficients zero.
  constexpr ARCCORE_HOST_DEVICE Real3x3()
  : x(Real3::zero())
  , y(Real3::zero())
  , z(Real3::zero())
  {}

  //! Constructs the matrix with rows (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3x3(Real3 ax, Real3 ay, Real3 az)
  : x(ax)
  , y(ay)
  , z(az)
  {}

  /*!
   * \brief Constructs the tensor ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
   * \deprecated Use Real3x3(Real3 x,Real3 y,Real3 z) instead.
   */
  ARCANE_DEPRECATED_116 Real3x3(Real ax, Real ay, Real az, Real bx, Real by, Real bz, Real cx, Real cy, Real cz)
  : x(ax, bx, cx)
  , y(ay, by, cy)
  , z(az, bz, cz)
  {}

  //! Constructs a triplet identical to \a f
  Real3x3(const Real3x3& f) = default;

  //! Constructs a triplet identical to \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(const Real3x3POD& f)
  : x(f.x)
  , y(f.y)
  , z(f.z)
  {}

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(Real v)
  {
    x = y = z = v;
  }

  //! Constructs the triplet ((av[0], av[1], av[2]), (av[3], av[4], av[5]), (av[6], av[7], av[8]))
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(ConstArrayView<Real> av)
  : x(av[0], av[1], av[2])
  , y(av[3], av[4], av[5])
  , z(av[6], av[7], av[8])
  {}

  //! Copy assignment operator
  Real3x3& operator=(const Real3x3& f) = default;

  //! Assigns the triplet (v,v,v) to the instance.
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator=(Real v)
  {
    x = y = z = v;
    return (*this);
  }

 public:

  Real3 x; //!< first element of the triplet
  Real3 y; //!< first element of the triplet
  Real3 z; //!< first element of the triplet

 public:

  //! Constructs the null tensor.
  constexpr ARCCORE_HOST_DEVICE static Real3x3 null() { return Real3x3(); }

  //! Constructs the zero matrix
  constexpr ARCCORE_HOST_DEVICE static Real3x3 zero() { return Real3x3(); }

  //! Constructs the identity matrix
  constexpr ARCCORE_HOST_DEVICE static Real3x3 identity() { return Real3x3(Real3(1.0, 0.0, 0.0), Real3(0.0, 1.0, 0.0), Real3(0.0, 0.0, 1.0)); }

  //! Constructs the matrix ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static Real3x3 fromColumns(Real ax, Real ay, Real az, Real bx, Real by, Real bz, Real cx, Real cy, Real cz)
  {
    return Real3x3(Real3(ax, bx, cx), Real3(ay, by, cy), Real3(az, bz, cz));
  }

  //! Constructs the matrix ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static Real3x3 fromLines(Real ax, Real bx, Real cx, Real ay, Real by, Real cy, Real az, Real bz, Real cz)
  {
    return Real3x3(Real3(ax, bx, cx), Real3(ay, by, cy), Real3(az, bz, cz));
  }

 public:

  //! Returns a copy of the matrix
  constexpr ARCCORE_HOST_DEVICE Real3x3 copy() const { return (*this); }

  //! Resets the coefficients of the matrix to zero.
  constexpr ARCCORE_HOST_DEVICE Real3x3& reset()
  {
    *this = zero();
    return (*this);
  }

  //! Assigns the rows (ax,ay,az) to the instance
  constexpr ARCCORE_HOST_DEVICE Real3x3& assign(Real3 ax, Real3 ay, Real3 az)
  {
    x = ax;
    y = ay;
    z = az;
    return (*this);
  }

  //! Copies matrix \a f
  constexpr ARCCORE_HOST_DEVICE Real3x3& assign(Real3x3 f)
  {
    x = f.x;
    y = f.y;
    z = f.z;
    return (*this);
  }

  //! Returns a view over the nine elements of the matrix.
  //! [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 9, &x.x };
  }

  //! Returns a constant view over the nine elements of the matrix.
  //! [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 9, &x.x };
  }

  /*!
   * \brief Reads the matrix from the stream \a i
   * The matrix is read in the form of three Real3s.
   */
  std::istream& assign(std::istream& i);

  //! Writes the triplet to the stream \a o readable by an assign()
  std::ostream& print(std::ostream& o) const;

  //! Writes the triplet to the stream \a o in the form (x,y,z)
  std::ostream& printXyz(std::ostream& o) const;

  //! Adds \a b to the triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& add(Real3x3 b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return (*this);
  }

  //! Subtracts \a b from the triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& sub(Real3x3 b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return (*this);
  }

  //! Adds \a b to each component of the triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& addSame(Real3 b)
  {
    x += b;
    y += b;
    z += b;
    return (*this);
  }

  //! Subtracts \a b from each component of the triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& subSame(Real3 b)
  {
    x -= b;
    y -= b;
    z -= b;
    return (*this);
  }

  //! Adds \a b to the triplet.
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator+=(Real3x3 b) { return add(b); }

  //! Subtracts \a b from the triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator-=(Real3x3 b) { return sub(b); }

  //! Multiplies each component of the matrix by the real \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
  }

  //! Divides each component of the matrix by the real \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
  }

  //! Creates a triplet that equals this triplet added to \a b
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator+(Real3x3 b) const { return Real3x3(x + b.x, y + b.y, z + b.z); }

  //! Creates a triplet that equals \a b subtracted from this triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator-(Real3x3 b) const { return Real3x3(x - b.x, y - b.y, z - b.z); }

  //! Creates a tensor opposite to the current tensor
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator-() const { return Real3x3(-x, -y, -z); }

  /*!
   * \brief Compares the current instance component by component to \a b.
   *
   * \retval true if this.x==b.x and this.y==b.y and this.z==b.z.
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real3x3 b) const
  {
    return x == b.x && y == b.y && z == b.z;
  }

  /*!
   * \brief Compares two triplets.
   * For the notion of equality, see operator==()
   * \retval true if the two triplets are different,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real3x3 b) const
  {
    return !operator==(b);
  }

  /*!
   * \brief Read-only access to the \a i-th (between 0 and 2 inclusive) row of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real3 operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Read-only access to the \a i-th (between 0 and 2 inclusive) row of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real3 operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Read-only access to the \a i-th row and \a j-th column.
   * \param i row number to return
   * \param j column number to return
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i, Integer j) const
  {
    ARCCORE_CHECK_AT(i, 3);
    ARCCORE_CHECK_AT(j, 3);
    return (&x)[i][j];
  }

  /*!
   * \brief Access to the \a i-th row (between 0 and 2 inclusive) of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real3& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Access to the \a i-th row (between 0 and 2 inclusive) of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real3& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Access to the \a i-th row and \a j-th column.
   * \param i row number to return
   * \param j column number to return
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i, Integer j)
  {
    ARCCORE_CHECK_AT(i, 3);
    ARCCORE_CHECK_AT(j, 3);
    return (&x)[i][j];
  }

  //! Determinant of the matrix
  constexpr ARCCORE_HOST_DEVICE Real determinant() const
  {
    return (x.x * (y.y * z.z - y.z * z.y) + x.y * (y.z * z.x - y.x * z.z) + x.z * (y.x * z.y - y.y * z.x));
  }

  //! Writes the triplet \a t to the stream \a o.
  friend std::ostream& operator<<(std::ostream& o, Real3x3 t)
  {
    return t.printXyz(o);
  }

  //! Reads the triplet \a t from the stream \a o.
  friend std::istream& operator>>(std::istream& i, Real3x3& t)
  {
    return t.assign(i);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator*(Real sca, Real3x3 vec)
  {
    return Real3x3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator*(Real3x3 vec, Real sca)
  {
    return Real3x3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator/(Real3x3 vec, Real sca)
  {
    return Real3x3(vec.x / sca, vec.y / sca, vec.z / sca);
  }

  /*!
  * \brief Comparison operator.
  *
  * This operator allows sorting Real3s for example
  * in std::set
  */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real3x3 v1, Real3x3 v2)
  {
    if (v1.x == v2.x) {
      if (v1.y == v2.y)
        return v1.z < v2.z;
      else
        return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

 public:

  // TODO: deprecate mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real3x3&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

 private:

  /*!
   * \brief Compares the values of \a a and \a b with the TypeEqualT comparator
   * \retval true if \a a and \a b are equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b)
  {
    return TypeEqualT<Real>::isEqual(a, b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
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
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
