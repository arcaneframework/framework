// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3.h                                                     (C) 2000-2026 */
/*                                                                           */
/* Vector of 3 dimensions of 'Real'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3_H
#define ARCANE_UTILS_REAL3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"
#include "arcane/utils/Real2.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct Real3POD
{
 public:

  Real x; //!< first component of the triplet
  Real y; //!< second component of the triplet
  Real z; //!< third component of the triplet

  /*!
   * Read-only access to the @a i-th component of Real3POD
   *
   * @note only works for x, y, and z ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * Read-only access to the @a i-th component of Real3POD
   *
   * @note only works for x, y, and z ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * Access to the @a i-th component of Real3POD
   *
   * @note only works for x, y, and z ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * Access to the @a i-th component of Real3POD
   *
   * @note only works for x, y, and z ordered in the POD
   *
   * @param i component number to return
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  //! Sets the \a i-th component to \a value
  ARCCORE_HOST_DEVICE void setComponent(Integer i, Real value)
  {
    ARCCORE_CHECK_AT(i, 3);
    (&x)[i] = value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a 3-dimensional real vector.

 Le vector includes three components \a x, \a y, and \a z, which are of type \b Real.

 \code
 Real3 value (1.0,2.3,4.5); // Created a triplet (x=1.0, y=2.3, z=4.5)
 cout << value.x;           // Prints component x 
 value.y += 3.2;            // Adds 3.2 to component \b y
 \endcode

 or equivalently

 \code
 Real3 value (1.0,2.3,4.5); // Created a triplet (x=1.0, y=2.3, z=4.5)
 cout << value[0];          // Prints component x 
 value[1] += 3.2;           // Adds 3.2 to component \b y
 \endcode
 */

class ARCANE_UTILS_EXPORT Real3
: public Real3POD
{
 public:

  //! Constructs the zero vector.
  constexpr ARCCORE_HOST_DEVICE Real3()
  : Real3POD()
  {
    x = 0.0;
    y = 0.0;
    z = 0.0;
  }

  //! Constructs the triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3(Real ax, Real ay, Real az)
  : Real3POD()
  {
    x = ax;
    y = ay;
    z = az;
  }

  //! Constructs a triplet identical to \a f
  Real3(const Real3& f) = default;

  //! Constructs a triplet identical to \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3(const Real3POD& f)
  : Real3POD()
  {
    x = f.x;
    y = f.y;
    z = f.z;
  }

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real3(Real v)
  : Real3POD()
  {
    x = y = z = v;
  }

  //! Constructs a triplet identical to \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3(const Real2& f)
  : Real3POD()
  {
    x = f.x;
    y = f.y;
    z = 0.0;
  }

  //! Constructs the triplet (av[0], av[1], av[2])
  constexpr ARCCORE_HOST_DEVICE Real3(ConstArrayView<Real> av)
  : Real3POD()
  {
    x = av[0];
    y = av[1];
    z = av[2];
  }

  //! Copy assignment operator.
  Real3& operator=(const Real3& f) = default;

  //! Assigns the triplet (v,v,v) to the instance.
  constexpr ARCCORE_HOST_DEVICE Real3& operator=(Real v)
  {
    x = y = z = v;
    return (*this);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE static Real3 null() { return Real3(0., 0., 0.); }
  constexpr ARCCORE_HOST_DEVICE static Real3 zero() { return Real3(0., 0., 0.); }

 public:

  //! Returns a copy of the triplet.
  constexpr ARCCORE_HOST_DEVICE Real3 copy() const { return (*this); }

  //! Resets the triplet using default constructors.
  constexpr ARCCORE_HOST_DEVICE Real3& reset()
  {
    x = y = z = 0.;
    return (*this);
  }

  //! Assigns the triplet (ax,ay,az) to the instance.
  constexpr ARCCORE_HOST_DEVICE Real3& assign(Real ax, Real ay, Real az)
  {
    x = ax;
    y = ay;
    z = az;
    return (*this);
  }

  //! Copies the triplet \a f
  constexpr ARCCORE_HOST_DEVICE Real3& assign(Real3 f)
  {
    x = f.x;
    y = f.y;
    z = f.z;
    return (*this);
  }

  //! Returns a view of the three elements of the vector.
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 3, &x };
  }

  //! Returns a constant view of the three elements of the vector.
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 3, &x };
  }

  //! Absolute value component by component.
  ARCCORE_HOST_DEVICE Real3 absolute() const { return Real3(math::abs(x), math::abs(y), math::abs(z)); }

  /*!
   * \brief Reads a triplet from the stream \a i
   * The triplet is read in the form of three values of type #value_type.
   */
  std::istream& assign(std::istream& i);

  //! Writes the triplet to the stream \a o readable by an assign()
  std::ostream& print(std::ostream& o) const;

  //! Writes the triplet to the stream \a o in the form (x,y,z)
  std::ostream& printXyz(std::ostream& o) const;

  //! Adds \a b to the triplet
  constexpr ARCCORE_HOST_DEVICE Real3& add(Real3 b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return (*this);
  }

  //! Subtracts \a b from the triplet
  constexpr ARCCORE_HOST_DEVICE Real3& sub(Real3 b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return (*this);
  }

  //! Multiplies each component of the triplet by the corresponding component of \a b
  constexpr ARCCORE_HOST_DEVICE Real3& mul(Real3 b)
  {
    x *= b.x;
    y *= b.y;
    z *= b.z;
    return (*this);
  }

  //! Divides each component of the triplet by the corresponding component of \a b
  constexpr ARCCORE_HOST_DEVICE Real3& div(Real3 b)
  {
    x /= b.x;
    y /= b.y;
    z /= b.z;
    return (*this);
  }

  //! Adds \a b to each component of the triplet
  constexpr ARCCORE_HOST_DEVICE Real3& addSame(Real b)
  {
    x += b;
    y += b;
    z += b;
    return (*this);
  }

  //! Subtracts b from each component of the triplet
  constexpr ARCCORE_HOST_DEVICE Real3& subSame(Real b)
  {
    x -= b;
    y -= b;
    z -= b;
    return (*this);
  }

  //! Multiplies each component of the triplet by b
  constexpr ARCCORE_HOST_DEVICE Real3& mulSame(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
    return (*this);
  }

  //! Divides each component of the triplet by b
  constexpr ARCCORE_HOST_DEVICE Real3& divSame(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
    return (*this);
  }

  //! Adds b to the triplet.
  constexpr ARCCORE_HOST_DEVICE Real3& operator+=(Real3 b) { return add(b); }

  //! Subtracts b from the triplet
  constexpr ARCCORE_HOST_DEVICE Real3& operator-=(Real3 b) { return sub(b); }

  //! Multiplies each component of the triplet by the corresponding component of b
  constexpr ARCCORE_HOST_DEVICE Real3& operator*=(Real3 b) { return mul(b); }

  //! Multiplies each component of the triplet by the real number b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
  }

  //! Divides each component of the triplet by the corresponding component of b
  constexpr ARCCORE_HOST_DEVICE Real3& operator/=(Real3 b) { return div(b); }

  //! Divides each component of the triplet by the real number b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
  }

  //! Creates a triplet that equals this triplet added to b
  constexpr ARCCORE_HOST_DEVICE Real3 operator+(Real3 b) const { return Real3(x + b.x, y + b.y, z + b.z); }

  //! Creates a triplet that equals b subtracted from this triplet
  constexpr ARCCORE_HOST_DEVICE Real3 operator-(Real3 b) const { return Real3(x - b.x, y - b.y, z - b.z); }

  //! Creates a triplet opposite to the current triplet
  constexpr ARCCORE_HOST_DEVICE Real3 operator-() const { return Real3(-x, -y, -z); }

  /*!
   * \brief Creates a triplet that equals this triplet whose each component has been
   * multiplied by the corresponding component of b.
   */
  constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real3 b) const { return Real3(x * b.x, y * b.y, z * b.z); }

  /*!
   * \brief Creates a triplet that equals this triplet whose each component has been divided
   * by the corresponding component of b.
   */
  constexpr ARCCORE_HOST_DEVICE Real3 operator/(Real3 b) const { return Real3(x / b.x, y / b.y, z / b.z); }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real sca, Real3 vec)
  {
    return Real3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Multiplication by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real3 vec, Real sca)
  {
    return Real3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Division by a scalar.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator/(Real3 vec, Real sca)
  {
    return Real3(vec.x / sca, vec.y / sca, vec.z / sca);
  }

 public:

  /*!
   * \brief Comparison operator.
   *
   * This operator allows sorting Real3 for example
   * in std::set
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real3 v1, Real3 v2)
  {
    if (v1.x == v2.x) {
      if (v1.y == v2.y)
        return v1.z < v2.z;
      else
        return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

  //! Writes the triplet t to the stream o
  friend std::ostream& operator<<(std::ostream& o, Real3 t)
  {
    return t.printXyz(o);
  }

  //! Reads the triplet t from the stream o.
  friend std::istream& operator>>(std::istream& i, Real3& t)
  {
    return t.assign(i);
  }

  /*!
   * \brief Compares the current instance component by component to b.
   *
   * \retval true if this.x==b.x and this.y==b.y and this.z==b.z.
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real3 b) const
  {
    return _eq(x, b.x) && _eq(y, b.y) && _eq(z, b.z);
  }

  /*!
   * \brief Compares two triplets.
   * For the notion of equality, see operator==()
   * \retval true if the two triplets are different,
   * \retval false otherwise.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real3 b) const { return !operator==(b); }

 public:

  //! Returns the square of the L2 norm of the triplet $\f$x^2+y^2+z^2$\f$
  // TODO: deprecate mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::squareNormL2(const Real3&) instead")
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const { return x * x + y * y + z * z; }

  //! Returns the L2 norm of the triplet $\f$\sqrt{x^2+y^2+z^2}\f$
  // TODO: deprecate mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::normL2(const Real3&) instead")
  inline ARCCORE_HOST_DEVICE Real normL2() const;

  //! Returns the square of the norm of the triplet $\f$x^2+y^2+z^2\f$
  ARCCORE_DEPRECATED_2021("Use math::squareNormL2(const Real3&) instead")
  constexpr ARCCORE_HOST_DEVICE Real abs2() const { return x * x + y * y + z * z; }

  //! Returns the norm of the triplet $\f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_DEPRECATED_2021("Use math::normL2(const Real3&) instead")
  inline ARCCORE_HOST_DEVICE Real abs() const;

  // TODO: deprecate mid-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real3&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

  // TODO: deprecate mid-2026: ARCANE_DEPRECATED_REASON("Y2024: Use math::mutableNormalize(Real3&) instead")
  inline Real3& normalize();

 private:

  /*!
   * \brief Compares the values of a and b using the TypeEqualT comparator
   * \retval true if a and b are equal,
   * \retval false otherwise.
   */
  inline constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b);

  //! Returns the square root of a
  inline ARCCORE_HOST_DEVICE static Real _sqrt(Real a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Real2::
Real2(const Real3& v)
: Real2POD()
{
  x = v.x;
  y = v.y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
  //! Returns the square of the L2 norm of the triplet $\f$x^2+y^2+z^2\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real3& v)
  {
    return v.x * v.x + v.y * v.y + v.z * v.z;
  }

  /*!
   * \brief Indicates if the instance is close to the zero instance.
   *
   * \retval true if math::isNearlyZero() is true for every component.
   * \retval false otherwise.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real3& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y) && math::isNearlyZero(v.z);
  }

  //! Returns the L2 norm of the triplet $\f$\sqrt{v.x^2+v.y^2+v.z^2}\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real3& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }

  /*!
    * \brief Normalizes the triplet v
    *
    * If the triplet is non-zero, divides each component by the norm of the triplet
    * (abs()), so that after calling this method, math::normL2() equals 1.
    * If the triplet is zero, does nothing.
    */
  inline Real3& mutableNormalize(Real3& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }

  /*!
    * \brief Returns the triplet v normalized with the L2 norm.
    *
    * If `math::normL2(v)` is non-zero, returns the triplet v divided by `math::normL2(v)`.
    * Otherwise, returns v.
    */
  inline Real3 normalizeL2(const Real3& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      return v / d;
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Real3& Real3::
normalize()
{
  return math::mutableNormalize(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real3::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real3::
_eq(Real a, Real b)
{
  return math::isEqual(a, b);
}

ARCCORE_HOST_DEVICE inline Real Real3::
_sqrt(Real a)
{
  return math::sqrt(a);
}

inline ARCCORE_HOST_DEVICE Real Real3::
normL2() const
{
  return math::normL2(*this);
}

inline ARCCORE_HOST_DEVICE Real Real3::
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
