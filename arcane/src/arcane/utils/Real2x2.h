// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2x2.h                                                   (C) 2000-2025 */
/*                                                                           */
/* 2x2 Matrix of 'Real'.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2X2_H
#define ARCANE_UTILS_REAL2X2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief POD structure for a Real2x2.
 */
struct Real2x2POD
{
 public:

  Real2POD x;
  Real2POD y;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a 2x2 matrix of reals.
 *
 * The matrix comprises two components \a x and \a y which are of the
 * type \b Real2. For example:
 *
 * \code
 * Real2x2 matrix;
 * matrix.x.y = 2.;
 * matrix.y.y = 3.;
 * \endcode
 */
class ARCANE_UTILS_EXPORT Real2x2
{
 public:

  //! Constructs the zero matrix
  constexpr ARCCORE_HOST_DEVICE Real2x2()
  : x(Real2::null())
  , y(Real2::null())
  {}

  //! Constructs the pair (ax,ay)
  constexpr ARCCORE_HOST_DEVICE Real2x2(Real2 ax, Real2 ay)
  : x(ax)
  , y(ay)
  {}

  /*!
   * \brief Constructs the pair ((ax,bx),(ay,by)).
   * \deprecated Use Real2x2(Real2 a,Real2 b) instead.
   */
  ARCANE_DEPRECATED_116 Real2x2(Real ax, Real ay, Real bx, Real by)
  : x(ax, bx)
  , y(ay, by)
  {}

  //! Constructs a copy identical to \a f
  Real2x2(const Real2x2& f) = default;

  //! Constructs a copy identical to \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real2x2(const Real2x2POD& f)
  : x(f.x)
  , y(f.y)
  {}

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real2x2(Real v)
  {
    x = y = v;
  }

  //! Constructs the pair ((av[0], av[1]), (av[2], av[3]))
  constexpr ARCCORE_HOST_DEVICE explicit Real2x2(ConstArrayView<Real> av)
  : x(av[0], av[1])
  , y(av[2], av[3])
  {}

  //! Copy assignment operator
  Real2x2& operator=(const Real2x2& f) = default;

  //! Assigns the pair (v,v,v) to the instance.
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator=(Real v)
  {
    x = y = v;
    return (*this);
  }

 public:

  Real2 x; //!< First component
  Real2 y; //!< Second component

 public:

  //! Constructs the zero matrix
  constexpr ARCCORE_HOST_DEVICE static Real2x2 null() { return Real2x2(); }

  //! Constructs the pair ((ax,bx),(ay,by)).
  constexpr ARCCORE_HOST_DEVICE static Real2x2 fromColumns(Real ax, Real ay, Real bx, Real by)
  {
    return Real2x2(Real2(ax, bx), Real2(ay, by));
  }

  //! Constructs the pair ((ax,bx),(ay,by)).
  constexpr ARCCORE_HOST_DEVICE static Real2x2 fromLines(Real ax, Real bx, Real ay, Real by)
  {
    return Real2x2(Real2(ax, bx), Real2(ay, by));
  }

 public:

  //! Returns a copy of the pair.
  constexpr ARCCORE_HOST_DEVICE Real2x2 copy() const { return (*this); }

  //! Resets the pair with default constructors.
  constexpr ARCCORE_HOST_DEVICE Real2x2& reset()
  {
    *this = null();
    return (*this);
  }

  //! Assigns the pair (ax,ay) to the instance
  constexpr ARCCORE_HOST_DEVICE Real2x2& assign(Real2 ax, Real2 ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }

  //! Copies the pair \a f
  constexpr ARCCORE_HOST_DEVICE Real2x2& assign(Real2x2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  //! Returns a view of the four elements of the matrix.
  //! [x.x, x.y, y.x, y.y]
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 4, &x.x };
  }

  //! Returns a constant view of the four elements of the matrix.
  //! [x.x, x.y, y.x, y.y]
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 4, &x.x };
  }

  /*!
   * \brief Reads the matrix from the stream \a i
   * The matrix is read in the form of three Real2.
   */
  std::istream& assign(std::istream& i);

  //! Writes the pair to the stream \a o readable by an assign()
  std::ostream& print(std::ostream& o) const;

  //! Writes the pair to the stream \a o in the form (x,y,z)
  std::ostream& printXy(std::ostream& o) const;

  //! Adds \a b to the pair
  constexpr ARCCORE_HOST_DEVICE Real2x2& add(Real2x2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }

  //! Subtracts \a b from the pair
  constexpr ARCCORE_HOST_DEVICE Real2x2& sub(Real2x2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }

  //! Multiplies each component of the pair by the corresponding component of \a b
  //Real2x2& mul(Real2x2 b) { x*=b.x; y*=b.y; return (*this); }

  //! Divides each component of the pair by the corresponding component of \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& div(Real2x2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }

  //! Adds \a b to each component of the pair
  constexpr ARCCORE_HOST_DEVICE Real2x2& addSame(Real2 b)
  {
    x += b;
    y += b;
    return (*this);
  }

  //! Subtracts \a b from each component of the pair
  constexpr ARCCORE_HOST_DEVICE Real2x2& subSame(Real2 b)
  {
    x -= b;
    y -= b;
    return (*this);
  }

  //! Multiplies each component of the pair by \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& mulSame(Real2 b)
  {
    x *= b;
    y *= b;
    return (*this);
  }

  //! Divides each component of the pair by \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& divSame(Real2 b)
  {
    x /= b;
    y /= b;
    return (*this);
  }

  //! Adds \a b to the pair.
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator+=(Real2x2 b) { return add(b); }

  //! Subtracts \a b from the pair
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator-=(Real2x2 b) { return sub(b); }

  //! Multiplies each component of the pair by the corresponding component of \a b
  //Real2x2& operator*=(Real2x2 b) { return mul(b); }

  //! Multiplies each component of the matrix by the real \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }

  //! Divides each component of the pair by the corresponding component of \a b
  //Real2x2& operator/= (Real2x2 b) { return div(b); }

  //! Divides each component of the matrix by the real \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }

  //! Creates a pair that equals this pair added to \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator+(Real2x2 b) const { return Real2x2(x + b.x, y + b.y); }

  //! Creates a pair that equals \a b subtracted from this pair
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator-(Real2x2 b) const { return Real2x2(x - b.x, y - b.y); }

  //! Creates an inverse tensor of the current tensor
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator-() const { return Real2x2(-x, -y); }

  /*!
   * \brief Compares component by component the current instance to \a b.
   *
   * \retval true if this.x==b.x and this.y==b.y.
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real2x2 b) const
  {
    return (x == b.x) && (y == b.y);
  }

  /*!
   * \brief Compares two pairs.
   * For the notion of equality, see operator==()
   * \retval true if the two pairs are different,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real2x2 b) const
  {
    return !operator==(b);
  }

  /*!
   * \brief Read-only access to the \a i-th (between 0 and 1 inclusive) row of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real2 operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Read-only access to the \a i-th (between 0 and 1 inclusive) row of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real2 operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Read-only access to the \a i-th row and \a j-th column.
   * \param i row number to return
   * \param j column number to return
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i, Integer j) const
  {
    ARCCORE_CHECK_AT(i, 2);
    ARCCORE_CHECK_AT(j, 2);
    return (&x)[i][j];
  }

  /*!
   * \brief Access to the \a i-th row (between 0 and 1 inclusive) of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real2& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Access to the \a i-th row (between 0 and 1 inclusive) of the instance.
   * \param i row number to return
   */
  ARCCORE_HOST_DEVICE Real2& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Access to the \a i-th row and \a j-th column.
   * \param i row number to return
   * \param j column number to return
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i, Integer j)
  {
    ARCCORE_CHECK_AT(i, 2);
    ARCCORE_CHECK_AT(j, 2);
    return (&x)[i][j];
  }

 public:

  //! Writes the pair \a t to the stream \a o
  friend std::ostream& operator<<(std::ostream& o, Real2x2 t)
  {
    return t.printXy(o);
  }

  //! Reads the pair \a t from the stream \a o.
  friend std::istream& operator>>(std::istream& i, Real2x2& t)
  {
    return t.assign(i);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator*(Real sca, Real2x2 vec)
  {
    return Real2x2(vec.x * sca, vec.y * sca);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator*(Real2x2 vec, Real sca)
  {
    return Real2x2(vec.x * sca, vec.y * sca);
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator/(Real2x2 vec, Real sca)
  {
    return Real2x2(vec.x / sca, vec.y / sca);
  }

  /*!
  * \brief Comparison operator.
  *
  * This operator allows sorting Real2x2 for example
  * in std::set
  */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real2x2 v1, Real2x2 v2)
  {
    if (v1.x == v2.x) {
      return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

 public:

  /*!
   * \brief Compares the matrix with the zero matrix.
   *
   * The matrix is zero if and only if each of its components
   * is less than a given epsilon. The epsilon value used is that
   * of float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon\f]
   *
   * \retval true if the matrix is equal to the zero matrix,
   * \retval false otherwise.
   */
  // TODO: make obsolete mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real2x2&) instead")
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
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon\f]
   *
   * \retval true if the matrix is equal to the zero matrix,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real2x2& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y);
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real2x2::
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
