// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3Proxy.h                                                (C) 2000-2018 */
/*                                                                           */
/* Proxy for a 'Real3'.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3PROXY_H
#define ARCANE_UTILS_REAL3PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/utils/BuiltInProxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef BuiltInProxy<Real> RealProxy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Proxy for a Real3.
 */
class ARCANE_UTILS_EXPORT Real3Proxy
: public Real3POD
{
 public:
 
  //! Constructs the triplet (ax,ay,az)
  Real3Proxy(Real3& value,const MemoryAccessInfo& info)
  : x(value.x,info), y(value.y,info), z(value.z,info), m_value(value), m_info(info) {}

  //! Constructs a triplet identical to \a f
  Real3Proxy(const Real3Proxy& f)
	: x(f.x), y(f.y), z(f.z), m_value(f.m_value), m_info(f.m_info) {}
  Real3 operator=(const Real3Proxy f)
    { x = f.x; y = f.y; z = f.z; return m_value; }
  Real3 operator=(Real3 f)
    { x = f.x; y = f.y; z = f.z; return m_value; }

  //! Assigns the triplet (v,v,v) to the instance.
  Real3 operator= (Real v)
    { x = v; y = v; z = v; return m_value; }
  operator Real3() const
    {
      return getValue();
    }
  //operator Real3&()
  //{
  //  return getValueMutable();
  //}
 public:

  RealProxy x; //!< first component of the triplet
  RealProxy y; //!< second component of the triplet
  RealProxy z; //!< third component of the triplet

 private:

  Real3& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Returns a copy of the triplet.
  Real3 copy() const { return Real3(x,y,z); }

  //! Resets the triplet with default constructors.
  Real3Proxy& reset() { x = y = z = 0.; return (*this); }

  //! Assigns the triplet (ax,ay,az) to the instance
  Real3Proxy& assign(Real ax,Real ay,Real az)
    { x = ax; y = ay; z = az; return (*this); }

  //! Copies the triplet \a f
  Real3Proxy& assign(Real3 f)
    { x = f.x; y = f.y; z = f.z; return (*this); }

  /*!
   * \brief Compares the triplet with the zero triplet.
   *
   * In the case of an integral type #value_type, the triplet is
   * zero if and only if each of its components is equal to 0.
   *
   * For #value_type of the floating-point type (float, double or #Real), the triplet
   * is zero if and only if each of its components is less
   * than a given epsilon. The value of the epsilon used is that
   * of float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
   *
   * \retval true if the triplet is equal to the zero triplet,
   * \retval false otherwise.
   */
  bool isNearlyZero() const
    {
      return math::isNearlyZero(x.getValue()) && math::isNearlyZero(y.getValue()) && math::isNearlyZero(z.getValue());
    }

  //! Returns the square of the norm of the triplet \f$x^2+y^2+z^2\f$
  Real abs2() const
    { return x*x + y*y + z*z; }

  //! Returns the norm of the triplet \f$\sqrt{x^2+y^2+z^2}\f$
  Real abs() const
    { return _sqrt(abs2()); }

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
  Real3Proxy& add(Real3 b) { x+=b.x; y+=b.y; z+=b.z; return (*this); }

  //! Subtracts \a b from the triplet
  Real3Proxy& sub(Real3 b) { x-=b.x; y-=b.y; z-=b.z; return (*this); }

  //! Multiplies each component of the triplet by the corresponding component of \a b
  Real3Proxy& mul(Real3 b) { x*=b.x; y*=b.y; z*=b.z; return (*this); }

  //! Divides each component of the triplet by the corresponding component of \a b
  Real3Proxy& div(Real3 b) { x/=b.x; y/=b.y; z/=b.z; return (*this); }

  //! Adds \a b to each component of the triplet
  Real3Proxy& addSame(Real b) { x+=b; y+=b; z+=b; return (*this); }

  //! Subtracts \a b from each component of the triplet
  Real3Proxy& subSame(Real b) { x-=b; y-=b; z-=b; return (*this); }

  //! Multiplies each component of the triplet by \a b
  Real3Proxy& mulSame(Real b) { x*=b; y*=b; z*=b; return (*this); }

  //! Divides each component of the triplet by \a b
  Real3Proxy& divSame(Real b) { x/=b; y/=b; z/=b; return (*this); }

  //! Adds \a b to the triplet.
  Real3Proxy& operator+= (Real3 b) { return add(b); }

  //! Subtracts \a b from the triplet
  Real3Proxy& operator-= (Real3 b) { return sub(b); }

  //! Multiplies each component of the triplet by the corresponding component of \a b
  Real3Proxy& operator*= (Real3 b) { return mul(b); }

  //! Multiplies each component of the triplet by the real number \a b
  void operator*= (Real  b) { x*=b; y*=b; z*=b; }

  //! Divides each component of the triplet by the corresponding component of \a b
  Real3Proxy& operator/= (Real3 b) { return div(b); }

  //! Divides each component of the triplet by the real number \a b
  void operator/= (Real  b) { x/=b; y/=b; z/=b; }

  //! Creates a triplet that equals this triplet added to \a b
  Real3 operator+  (Real3 b)  const { return Real3(x+b.x,y+b.y,z+b.z); }

  //! Creates a triplet that equals \a b subtracted from this triplet
  Real3 operator-  (Real3 b)  const { return Real3(x-b.x,y-b.y,z-b.z); }

  //! Creates a triplet opposite to the current triplet
  Real3 operator-() const { return Real3(-x,-y,-z); }

  /*!
   * \brief Creates a triplet that equals this triplet whose each component has been
   * multiplied by the corresponding component of \a b.
   */
  Real3 operator*(Real3 b) const { return Real3(x*b.x,y*b.y,z*b.z); }

  /*!
   * \brief Creates a triplet that equals this triplet whose each component has been divided
   * by the corresponding component of \a b.
   */
  Real3 operator/(Real3 b) const { return Real3(x/b.x,y/b.y,z/b.z); }
										
  /*!
   * \brief Normalizes the triplet.
   * 
   * If the triplet is non-zero, divides each component by the norm of the triplet
   * (abs()), so that after calling this method, abs() equals \a 1.
   * If the triplet is zero, does nothing.
   */
  Real3Proxy& normalize()
    {
      Real d = abs();
      if (!math::isZero(d))
        divSame(d);
      return (*this);
    }
	
  /*!
   * \brief Compares the triplet to \a b.
   *
   * In the case of an integral type #value_type, two triplets
   * are equal if and only if each of their components are strictly
   * equal.
   *
   * For #value_type of the floating-point type (float, double or #Real), two triplets
   * are identical if and only if the absolute value of the difference
   * between each of their corresponding components is less
   * than a given epsilon. The value of the epsilon used is that
   * of float_info<value_type>::nearlyEpsilon():
   * \f[A=B \Leftrightarrow |A.x-B.x|<\epsilon,|A.y-B.y|<\epsilon,|A.z-B.z|<\epsilon \f]
   * \retval true if the two triplets are equal,
   * \retval false otherwise.
   */
  friend bool operator==(Real3Proxy a,Real3Proxy b)
  { return  _eq(a.x,b.x) &&  _eq(a.y,b.y) && _eq(a.z,b.z); }

  /*!
   * \brief Compares two triplets.
   * For the notion of equality, see operator==()
   * \retval true if the two triplets are different,
   * \retval false otherwise.
   */
  friend bool operator!=(Real3Proxy a,Real3Proxy b)
  { return !(a==b); }

 public:
  Real3 getValue() const
    {
      m_info.setRead();
      return m_value;
    }
  Real3& getValueMutable()
    {
      m_info.setReadOrWrite();
      return m_value;
    }
 private:

  /*!
   * \brief Compares the values of \a a and \a b using the TypeEqualT comparator
   * \retval true if \a a and \a b are equal,
   * \retval false otherwise.
   */
  static bool _eq(Real a,Real b)
    { return math::isEqual(a,b); }

  //! Returns the square root of \a a
  static Real _sqrt(Real a)
    { return math::sqrt(a); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplication by a scalar.
 */
inline Real3 operator*(Real sca,Real3Proxy vec)
{
  return Real3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multiplication by a scalar.
 */
inline Real3
operator*(Real3Proxy vec,Real sca)
{
  return Real3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Division by a scalar.
 */
inline Real3
operator/(Real3Proxy vec,Real sca)
{
  return Real3(vec.x/sca,vec.y/sca,vec.z/sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Comparison operator.
 *
 * This operator allows Real3s to be sorted for example
 * in std::set
 */
inline bool
operator<(const Real3Proxy v1,const Real3Proxy v2)
{
  return v1.getValue()<v2.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the triplet \a t to the stream \a o
 * \relates Real3
 */
inline std::ostream&
operator<< (std::ostream& o,Real3Proxy t)
{
  return t.printXyz(o);
}

/*!
 * \brief Reads the triplet \a t from the stream \a o.
 * \relates Real3
 */
inline std::istream&
operator>> (std::istream& i,Real3Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
