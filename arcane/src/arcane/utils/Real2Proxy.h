// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2Proxy.h                                                (C) 2000-2008 */
/*                                                                           */
/* Proxy of a 'Real2'.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2PROXY_H
#define ARCANE_UTILS_REAL2PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/BuiltInProxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a 2-dimensional real vector.
 
 The vector comprises two components \a x and \a y which are of type \b Real.

 \code
 Real2Proxy value (1.0,2.3); // Created a pair (x=1.0, y=2.3)
 cout << value.x;   // Prints the x component 
 value.y += 3.2; // Adds 3.2 to the \b y component
 \endcode
 */
class ARCANE_UTILS_EXPORT Real2Proxy
{
 public:

  //! Constructs the pair (ax,ay)
  Real2Proxy(Real2& value, const MemoryAccessInfo& info)
  : x(value.x, info)
  , y(value.y, info)
  , m_value(value)
  , m_info(info)
  {}

  //! Constructs a pair identical to \a f
  Real2Proxy(const Real2Proxy& f)
  : x(f.x)
  , y(f.y)
  , m_value(f.m_value)
  , m_info(f.m_info)
  {}
  const Real2& operator=(Real2Proxy f)
  {
    x = f.x;
    y = f.y;
    return m_value;
  }
  const Real2& operator=(Real2 f)
  {
    x = f.x;
    y = f.y;
    return m_value;
  }

  //! Assigns the pair (v,v) to the instance.
  const Real2& operator=(Real v)
  {
    x = y = v;
    return m_value;
  }
  operator Real2() const
  {
    return getValue();
  }
  //operator Real2&()
  //{
  //  return getValueMutable();
  //}
 public:

  RealProxy x; //!< first component of the pair
  RealProxy y; //!< second component of the pair

 private:

  Real2& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Returns a copy of the pair.
  Real2 copy() const { return m_value; }

  //! Resets the pair with default constructors.
  Real2Proxy& reset()
  {
    x = y = 0.;
    return (*this);
  }

  //! Assigns the pair (ax,ay,az) to the instance
  Real2Proxy& assign(Real ax, Real ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }

  //! Copies the pair \a f
  Real2Proxy& assign(Real2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  /*!
   * \brief Compares the pair with the zero pair.
   *
   * In the case of an integral #value_type, the pair is
   * zero if and only if each of its components is equal to 0.
   *
   * For #value_type of the floating point type (float, double or #Real), the pair
   * is zero if and only if each of its components is less
   * than a given epsilon. The value of the epsilon used is that
   * of float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon \f]
   *
   * \retval true if the pair is equal to the zero pair,
   * \retval false otherwise.
   */
  bool isNearlyZero() const
  {
    return math::isNearlyZero(x.getValue()) && math::isNearlyZero(y.getValue());
  }

  //! Returns the squared norm of the pair \f$x^2+y^2+z^2\f$
  Real abs2() const
  {
    return x * x + y * y;
  }

  //! Returns the norm of the pair \f$\sqrt{x^2+y^2+z^2}\f$
  Real abs() const
  {
    return _sqrt(abs2());
  }

  /*!
   * \brief Reads a pair from the stream \a i
   * The pair is read in the form of three #value_type values.
   */
  istream& assign(istream& i);

  //! Writes the pair to the stream \a o readable by an assign()
  ostream& print(ostream& o) const;

  //! Writes the pair to the stream \a o in the form (x,y)
  ostream& printXy(ostream& o) const;

  //! Adds \a b to the pair
  Real2Proxy& add(Real2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }

  //! Subtracts \a b from the pair
  Real2Proxy& sub(Real2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }

  //! Multiplies each component of the pair by the corresponding component of \a b
  Real2Proxy& mul(Real2 b)
  {
    x *= b.x;
    y *= b.y;
    return (*this);
  }

  //! Divides each component of the pair by the corresponding component of \a b
  Real2Proxy& div(Real2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }

  //! Adds \a b to each component of the pair
  Real2Proxy& addSame(Real b)
  {
    x += b;
    y += b;
    return (*this);
  }

  //! Subtracts \a b from each component of the pair
  Real2Proxy& subSame(Real b)
  {
    x -= b;
    y -= b;
    return (*this);
  }

  //! Multiplies each component of the pair by \a b
  Real2Proxy& mulSame(Real b)
  {
    x *= b;
    y *= b;
    return (*this);
  }

  //! Divides each component of the pair by \a b
  Real2Proxy& divSame(Real b)
  {
    x /= b;
    y /= b;
    return (*this);
  }

  //! Adds \a b to the pair.
  Real2Proxy& operator+=(Real2 b) { return add(b); }

  //! Subtracts \a b from the pair
  Real2Proxy& operator-=(Real2 b) { return sub(b); }

  //! Multiplies each component of the pair by the corresponding component of \a b
  Real2Proxy& operator*=(Real2 b) { return mul(b); }

  //! Multiplies each component of the pair by the real \a b
  void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }

  //! Divides each component of the pair by the corresponding component of \a b
  Real2Proxy& operator/=(Real2 b) { return div(b); }

  //! Divides each component of the pair by the real \a b
  void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }

  //! Creates a pair that equals this pair added to \a b
  Real2 operator+(Real2 b) const { return Real2(x + b.x, y + b.y); }

  //! Creates a pair that equals \a b subtracted from this pair
  Real2 operator-(Real2 b) const { return Real2(x - b.x, y - b.y); }

  //! Creates a pair opposite to the current pair
  Real2 operator-() const { return Real2(-x, -y); }

  /*!
   * \brief Creates a pair that equals this pair where each component has been
   * multiplied by the corresponding component of \a b.
   */
  Real2 operator*(Real2 b) const { return Real2(x * b.x, y * b.y); }

  /*!
   * \brief Creates a pair that equals this pair where each component has been divided
   * by the corresponding component of \a b.
   */
  Real2 operator/(Real2 b) const { return Real2(x / b.x, y / b.y); }

  /*!
   * \brief Normalizes the pair.
   *
   * If the pair is non-zero, divides each component by the norm of the pair
   * (abs()), such that after calling this method, abs() equals \a 1.
   * If the pair is zero, does nothing.
   */
  Real2Proxy& normalize()
  {
    Real d = abs();
    if (!math::isZero(d))
      divSame(d);
    return (*this);
  }

  /*!
   * \brief Compares the pair to \a b.
   *
   * In the case of an integral #value_type, two pairs
   * are equal if and only if each of their components are strictly
   * equal.
   *
   * For #value_type of the floating point type (float, double or #Real), two pairs
   * are identical if and only if the absolute value of the difference
   * between each of their corresponding components is less
   * than a given epsilon. The value of the epsilon used is that
   * of float_info<value_type>::nearlyEpsilon():
   * \f[A=B \Leftrightarrow |A.x-B.x|<\epsilon,|A.y-B.y|<\epsilon,|A.z-B.z|<\epsilon \f]
   * \retval true if the two pairs are equal,
   * \retval false otherwise.
   */
  bool operator==(Real2 b) const
  {
    return _eq(x, b.x) && _eq(y, b.y);
  }

  /*!
   * \brief Compares two pairs.
   * For the notion of equality, see operator==()
   * \retval true if the two pairs are different,
   * \retval false otherwise.
   */
  bool operator!=(Real2 b) const
  {
    return !operator==(b);
  }

 public:

  Real2 getValue() const
  {
    m_info.setRead();
    return m_value;
  }
  Real2& getValueMutable()
  {
    m_info.setReadOrWrite();
    return m_value;
  }

 private:

  /*!
   * \brief Compares the values of \a a and \a b with the TypeEqualT comparator
   * \retval true if \a a and \a b are equal,
   * \retval false otherwise.
   */
  static bool _eq(Real a, Real b)
  {
    return math::isEqual(a, b);
  }

  //! Returns the square root of \a a
  static Real _sqrt(Real a)
  {
    return math::sqrt(a);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplication by a scalar.
 */
inline Real2 operator*(Real sca, const Real2Proxy& vec)
{
  return Real2(vec.x * sca, vec.y * sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplication by a scalar.
 */
inline Real2
operator*(const Real2Proxy& vec, Real sca)
{
  return Real2(vec.x * sca, vec.y * sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Division by a scalar.
 */
inline Real2
operator/(const Real2Proxy& vec, Real sca)
{
  return Real2(vec.x / sca, vec.y / sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Comparison operator.
 *
 * This operator allows sorting Real2Proxy for example
 * in std::set
 */
inline bool
operator<(const Real2Proxy& v1, const Real2Proxy& v2)
{
  return v1.getValue() < v2.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the pair \a t to the stream \a o
 * \relates Real2Proxy
 */
inline ostream&
operator<<(ostream& o, Real2Proxy t)
{
  return t.printXy(o);
}

/*!
 * \brief Reads the pair \a t from the stream \a o.
 * \relates Real2Proxy
 */
inline istream&
operator>>(istream& i, Real2Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
