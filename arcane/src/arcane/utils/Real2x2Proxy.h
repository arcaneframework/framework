// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2x2Proxy.h                                              (C) 2000-2008 */
/*                                                                           */
/* Proxy of type 'Real2x2'.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2X2PROXY_H
#define ARCANE_UTILS_REAL2X2PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real2Proxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Proxy of type 'Real2x2'.
 */
class ARCANE_UTILS_EXPORT Real2x2Proxy
{
 public:

  //! Constructs the pair (ax,ay)
  Real2x2Proxy(Real2x2& value, const MemoryAccessInfo& info)
  : x(value.x, info)
  , y(value.y, info)
  , m_value(value)
  , m_info(info)
  {}

  //! Constructs a pair identical to \a f
  Real2x2Proxy(const Real2x2Proxy& f)
  : x(f.x)
  , y(f.y)
  , m_value(f.m_value)
  , m_info(f.m_info)
  {}
  const Real2x2Proxy& operator=(const Real2x2Proxy& f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }
  const Real2x2Proxy& operator=(Real2x2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  //! Assigns the triplet (v,v,v) to the instance.
  const Real2x2Proxy& operator=(Real v)
  {
    x = v;
    y = v;
    return (*this);
  }

 public:

  Real2Proxy x; //!< First component
  Real2Proxy y; //!< Second component

 private:

  Real2x2& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Returns a copy of the pair.
  Real2x2 copy() const { return getValue(); }

  //! Resets the pair using default constructors.
  Real2x2Proxy& reset()
  {
    *this = Real2x2::null();
    return (*this);
  }

  //! Assigns the triplet (ax,ay,az) to the instance
  Real2x2Proxy& assign(Real2 ax, Real2 ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }

  //! Copies the pair \a f
  Real2x2Proxy& assign(Real2x2Proxy f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }
  operator const Real2x2&() const
  {
    return getValue();
  }
  //operator Real2x2&()
  //{
  //  return getValueMutable();
  //}

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
  bool isNearlyZero() const
  {
    return x.isNearlyZero() && y.isNearlyZero();
  }

  /*!
   * \brief Reads the matrix from the stream \a i
   * The matrix is read in the form of three Real2s.
   */
  istream& assign(istream& i);

  //! Writes the pair to the stream \a o readable by an assign()
  ostream& print(ostream& o) const;

  //! Writes the pair to the stream \a o in the form (x,y,z)
  ostream& printXy(ostream& o) const;

  //! Adds \a b to the pair
  Real2x2Proxy& add(Real2x2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }

  //! Subtracts \a b from the pair
  Real2x2Proxy& sub(Real2x2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }

  //! Multiplies each component of the pair by the corresponding component of \a b
  //Real2x2Proxy& mul(Real2x2Proxy b) { x*=b.x; y*=b.y; return (*this); }

  //! Divides each component of the pair by the corresponding component of \a b
  Real2x2Proxy& div(Real2x2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }

  //! Adds \a b to each component of the pair
  Real2x2Proxy& addSame(Real2 b)
  {
    x += b;
    y += b;
    return (*this);
  }

  //! Subtracts \a b from each component of the pair
  Real2x2Proxy& subSame(Real2 b)
  {
    x -= b;
    y -= b;
    return (*this);
  }

  //! Multiplies each component of the pair by \a b
  Real2x2Proxy& mulSame(Real2 b)
  {
    x *= b;
    y *= b;
    return (*this);
  }

  //! Divides each component of the pair by \a b
  Real2x2Proxy& divSame(Real2 b)
  {
    x /= b;
    y /= b;
    return (*this);
  }

  //! Adds \a b to the pair.
  Real2x2Proxy& operator+=(Real2x2 b) { return add(b); }

  //! Subtracts \a b from the pair
  Real2x2Proxy& operator-=(Real2x2 b) { return sub(b); }

  //! Multiplies each component of the pair by the corresponding component of \a b
  //Real2x2Proxy& operator*=(Real2x2Proxy b) { return mul(b); }

  //! Multiplies each component of the matrix by the real \a b
  void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }

  //! Divides each component of the pair by the corresponding component of \a b
  //Real2x2Proxy& operator/= (Real2x2 b) { return div(b); }

  //! Divides each component of the matrix by the real \a b
  void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }

  //! Creates a pair that equals this pair added to \a b
  Real2x2 operator+(Real2x2 b) const { return Real2x2(x + b.x, y + b.y); }

  //! Creates a pair that equals \a b subtracted from this pair
  Real2x2 operator-(Real2x2 b) const { return Real2x2(x - b.x, y - b.y); }

  //! Creates a tensor opposite to the current tensor
  Real2x2 operator-() const { return Real2x2(-x, -y); }

  /*!
   * \brief Creates a pair that equals this pair whose each component has been
   * multiplied by the corresponding component of \a b.
   */
  //Real2x2 operator*(Real2x2Proxy b) const { return Real2x2Proxy(x*b.x,y*b.y); }

  /*!
   * \brief Creates a pair that equals this pair whose each component has been divided
   * by the corresponding component of \a b.
   */
  //Real2x2Proxy operator/(Real2x2Proxy b) const { return Real2x2Proxy(x/b.x,y/b.y); }

 public:

  const Real2x2& getValue() const
  {
    m_info.setRead();
    return m_value;
  }
  Real2x2& getValueMutable()
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
    return TypeEqualT<Real>::isEqual(a, b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(const Real2x2& a, const Real2x2Proxy& b)
{
  return a == b.getValue();
}
inline bool
operator==(const Real2x2Proxy& a, const Real2x2& b)
{
  return a.getValue() == b;
}
inline bool
operator==(const Real2x2Proxy& a, const Real2x2Proxy& b)
{
  return a.getValue() == b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator!=(const Real2x2& a, const Real2x2Proxy& b)
{
  return a != b.getValue();
}
inline bool
operator!=(const Real2x2Proxy& a, const Real2x2& b)
{
  return a.getValue() != b;
}
inline bool
operator!=(const Real2x2Proxy& a, const Real2x2Proxy& b)
{
  return a.getValue() != b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the pair \a t to the stream \a o
 * \relates Real2x2Proxy
 */
inline ostream&
operator<<(ostream& o, const Real2x2Proxy& t)
{
  return t.printXy(o);
}

/*!
 * \brief Reads the pair \a t from the stream \a o.
 * \relates Real2x2Proxy
 */
inline istream&
operator>>(istream& i, Real2x2Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Multiplication by a scalar. */
inline Real2x2
operator*(Real sca, const Real2x2Proxy& vec)
{
  return Real2x2(vec.x * sca, vec.y * sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Multiplication by a scalar. */
inline Real2x2
operator*(const Real2x2Proxy& vec, Real sca)
{
  return Real2x2(vec.x * sca, vec.y * sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Division by a scalar. */
inline Real2x2
operator/(const Real2x2Proxy& vec, Real sca)
{
  return Real2x2(vec.x / sca, vec.y / sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Comparison operator.
 *
 * This operator allows Real2s to be sorted for example
 * in std::set
 */
inline bool
operator<(const Real2x2Proxy& v1, const Real2x2Proxy& v2)
{
  return (v1.getValue() < v2.getValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
