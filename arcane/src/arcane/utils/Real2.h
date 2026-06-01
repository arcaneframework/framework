// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2.h                                                     (C) 2000-2026 */
/*                                                                           */
/* 2-dimensional vector of 'Real'.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2_H
#define ARCANE_UTILS_REAL2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct Real2POD
{
 public:

  Real x; //!< first component of the pair
  Real y; //!< second component of the pair

  /*!
   * Read-only access to the @a i th component of the Real2POD
   *
   * @note only works for x, y ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Read-only access to the @a i th component of the Real2POD
   *
   * @note only works for x, y ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Access to the @a i th component of the Real2POD
   *
   * @note only works for x, y ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Access to the @a i th component of the Real2POD
   *
   * @note only works for x, y ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  //! Positions the \a i th component at \a value
  ARCCORE_HOST_DEVICE void setComponent(Integer i, Real value)
  {
    ARCCORE_CHECK_AT(i, 2);
    (&x)[i] = value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a 2-dimensional real vector.

 The vector includes two components \a x and \a y which are of
 type \b Real.

 \code
 Real2 value (1.0,2.3); // Creates a pair (x=1.0, y=2.3)
 cout << value.x;   // Prints the x component
 value.y += 3.2; // Adds 3.2 to the \b y component
 \endcode
 */
class ARCANE_UTILS_EXPORT Real2
: public Real2POD
{
 public:

  //! Constructs the zero vector.
  constexpr ARCCORE_HOST_DEVICE Real2()
  : Real2POD()
  {
    x = 0.;
    y = 0.;
  }
  //! Constructs the pair (ax,ay)
  constexpr ARCCORE_HOST_DEVICE Real2(Real ax, Real ay)
  : Real2POD()
  {
    x = ax;
    y = ay;
  }
  //! Constructs a copy identical to \a f
  Real2(const Real2& f) = default;
  //! Constructs a copy identical to \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real2(const Real2POD& f)
  : Real2POD()
  {
    x = f.x;
    y = f.y;
  }

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real2(Real v)
  : Real2POD()
  {
    x = y = v;
  }

  //! Constructs the instance using the first two components of Real3.
  inline constexpr ARCCORE_HOST_DEVICE explicit Real2(const Real3& v);

  //! Constructs the pair (av[0], av[1])
  constexpr ARCCORE_HOST_DEVICE Real2(ConstArrayView<Real> av)
  : Real2POD()
  {
    x = av[0];
    y = av[1];
  }

  Real2& operator=(const Real2& f) = default;

  //! Assigns the pair (v,v) to the instance.
  constexpr ARCCORE_HOST_DEVICE Real2& operator=(Real v)
  {
    x = y = v;
    return (*this);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE static Real2 null() { return Real2(0., 0.); }

 public:

  //! Returns a copy of the pair.
  constexpr ARCCORE_HOST_DEVICE Real2 copy() const { return (*this); }

  //! Resets the pair using default constructors.
  constexpr ARCCORE_HOST_DEVICE Real2& reset()
  {
    x = y = 0.0;
    return (*this);
  }

  //! Assigns the triplet (ax,ay,az) to the instance
  constexpr ARCCORE_HOST_DEVICE Real2& assign(Real ax, Real ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }

  //! Copies the pair \a f
  constexpr ARCCORE_HOST_DEVICE Real2& assign(Real2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  //! Returns a view of the two elements of the vector.
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 2, &x };
  }

  //! Returns a constant view of the two elements of the vector.
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 2, &x };
  }

  //! Absolute value component by component.
  ARCCORE_HOST_DEVICE Real2 absolute() const { return Real2(math::abs(x), math::abs(y)); }

  /*!
   * \brief Reads a pair from stream \a i
   * The pair is read in the form of three values of type #value_type.
   */
  std::istream& assign(std::istream& i);

  //! Writes the pair to stream \a o readable by an assign()
  std::ostream& print(std::ostream& o) const;

  //! Writes the pair to stream \a o in the form (x,y)
  std::ostream& printXy(std::ostream& o) const;

  //! Adds \a b to the pair
  constexpr ARCCORE_HOST_DEVICE Real2& add(Real2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }

  //! Subtracts \a b from the pair
  constexpr ARCCORE_HOST_DEVICE Real2& sub(Real2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }

  //! Multiplies each component of the pair by the corresponding component of \a b
  constexpr ARCCORE_HOST_DEVICE Real2& mul(Real2 b)
  {
    x *= b.x;
    y *= b.y;
    return (*this);
  }

  //! Divides each component of the pair by the corresponding component of \a b
  constexpr ARCCORE_HOST_DEVICE Real2& div(Real2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }

  //! Adds \a b to each component of the pair
  constexpr ARCCORE_HOST_DEVICE Real2& addSame(Real b)
  {
    x += b;
    y += b;
    return (*this);
  }

  //! Subtracts \a b from each component of the pair
  constexpr ARCCORE_HOST_DEVICE Real2& subSame(Real b)
  {
    x -= b;
    y -= b;
    return (*this);
  }

  //! Multiplies each component of the pair by b
  constexpr ARCCORE_HOST_DEVICE Real2& mulSame(Real b)
  {
    x *= b;
    y *= b;
    return (*this);
  }

  //! Divides each component of the pair by b
  constexpr ARCCORE_HOST_DEVICE Real2& divSame(Real b)
  {
    x /= b;
    y /= b;
    return (*this);
  }

  //! Adds b to the pair.
  constexpr ARCCORE_HOST_DEVICE Real2& operator+=(Real2 b) { return add(b); }

  //! Subtracts b from the pair
  constexpr ARCCORE_HOST_DEVICE Real2& operator-=(Real2 b) { return sub(b); }

  //! Multiplies each component of the pair by the corresponding component of b
  constexpr ARCCORE_HOST_DEVICE Real2& operator*=(Real2 b) { return mul(b); }

  //! Multiplies each component of the pair by the real number b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }

  //! Divides each component of the pair by the corresponding component of b
  constexpr ARCCORE_HOST_DEVICE Real2& operator/=(Real2 b) { return div(b); }

  //! Divides each component of the pair by the real number b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }

  //! Creates a pair that equals this pair added to b
  constexpr ARCCORE_HOST_DEVICE Real2 operator+(Real2 b) const { return Real2(x + b.x, y + b.y); }

  //! Creates a pair that equals b subtracted from this pair
  constexpr ARCCORE_HOST_DEVICE Real2 operator-(Real2 b) const { return Real2(x - b.x, y - b.y); }

  //! Creates a pair opposite to the current pair
  constexpr ARCCORE_HOST_DEVICE Real2 operator-() const { return Real2(-x, -y); }

  /*!
   * \brief Creates a pair that equals this pair, where each component has been
   * multiplied by the corresponding component of b.
   */
  constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real2 b) const { return Real2(x * b.x, y * b.y); }

  /*!
   * \brief Creates a pair that equals this pair, where each component has been divided
   * by the corresponding component of b.
   */
  constexpr ARCCORE_HOST_DEVICE Real2 operator/(Real2 b) const { return Real2(x / b.x, y / b.y); }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real sca, Real2 vec)
  {
    return Real2(vec.x * sca, vec.y * sca);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real2 vec, Real sca)
  {
    return Real2(vec.x * sca, vec.y * sca);
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator/(Real2 vec, Real sca)
  {
    return Real2(vec.x / sca, vec.y / sca);
  }

  /*!
   * \brief Comparison operator.
   *
   * This operator allows Real2 objects to be sorted for use, for example,
   * in std::set
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real2 v1, Real2 v2)
  {
    if (v1.x == v2.x) {
      return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

  //! Writes the pair t to the stream o.
  friend std::ostream& operator<<(std::ostream& o, Real2 t)
  {
    return t.printXy(o);
  }

  //! Reads the pair t from the stream o.
  friend std::istream& operator>>(std::istream& i, Real2& t)
  {
    return t.assign(i);
  }

  /*!
   * \brief Compares the current instance component by component to b.
   *
   * \retval true if this.x==b.x and this.y==b.y.
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real2 b) const
  {
    return _eq(x, b.x) && _eq(y, b.y);
  }

  /*!
   * \brief Compares two pairs.
   * For the concept of equality, see operator==()
   * \retval true if the two pairs are different,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real2 b) const { return !operator==(b); }

 public:

  //! Returns the squared norm of the pair $\f$x^2+y^2+z^2$\f$
  // TODO: make obsolete mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::squareNormL2(*this) instead")
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const { return x * x + y * y; }

  //! Returns the squared norm of the pair $\f$x^2+y^2+z^2$\f$
  ARCCORE_DEPRECATED_2021("Use math::squareNormL2(*this) instead")
  ARCCORE_HOST_DEVICE Real abs2() const { return x * x + y * y; }

  //! Returns the norm of the pair $\f$\sqrt{x^2+y^2+z^2}$\f$
  ARCCORE_DEPRECATED_2021("Use math::normL2(*this) instead")
  inline ARCCORE_HOST_DEVICE Real abs() const;

  /*!
   * \brief Indicates if the instance is close to the zero instance.
   *
   * \retval true if math::isNearlyZero() is true for every component.
   * \retval false otherwise.
   */
  // TODO: make obsolete mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real2&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

  //! Returns the norm of the pair $\f$\sqrt{x^2+y^2+z^2}$\f$
  // TODO: make obsolete mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::normL2(const Real2&) instead")
  ARCCORE_HOST_DEVICE Real normL2() const;

  // TODO: make obsolete mid-2026 ARCANE_DEPRECATED_REASON("Y2026: Use math::mutableNormalize(Real2&) instead")
  /*!
   * \brief Normalizes the pair.
   *
   * If the pair is non-zero, divides each component by the norm of the pair
   * (abs()), such that after calling this method, abs() equals 1.
   * If the pair is zero, does nothing.
   */
  inline Real2& normalize();

 private:

  /*!
   * \brief Compares the values of a and b using the TypeEqualT comparator
   * \retval true if a and b are equal,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b);

  //! Returns the square root of a
  ARCCORE_HOST_DEVICE static Real _sqrt(Real a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
  /*!
   * \brief Indicates if the instance is close to the zero instance.
   *
   * \retval true if math::isNearlyZero() is true for every component.
   * \retval false otherwise.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real2& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y);
  }

  //! Returns the squared norm of the pair $\f$x^2+y^2+z^2$\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real2& v)
  {
    return v.x * v.x + v.y * v.y;
  }

  //! Returns the norm of the pair $\f$\sqrt{x^2+y^2+z^2}$\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real2& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }

  /*!
   * \brief Normalizes the pair.
   *
   * If the pair is non-zero, divides each component by the norm of the pair
   * (abs()), such that after calling this method, abs() equals 1.
   * If the pair is zero, does nothing.
   */
  inline Real2& mutableNormalize(Real2& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }

  /*!
    * \brief Returns the pair v normalized by the L2 norm.
    *
    * If `math::normL2(v)` is non-zero, returns the pair v divided by `math::normL2(v)`.
    * Otherwise, returns v.
    */
  inline Real2 normalizeL2(const Real2& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      return v / d;
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real2::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real2::
_eq(Real a, Real b)
{
  return math::isEqual(a, b);
}

inline ARCCORE_HOST_DEVICE Real Real2::
_sqrt(Real a)
{
  return math::sqrt(a);
}

inline ARCCORE_HOST_DEVICE Real Real2::
normL2() const
{
  return math::normL2(*this);
}

inline Real2& Real2::
normalize()
{
  return math::mutableNormalize(*this);
}

inline ARCCORE_HOST_DEVICE Real Real2::
abs() const
{
  return math::normL2(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
