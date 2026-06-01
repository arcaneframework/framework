// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3x3Proxy.h                                              (C) 2000-2008 */
/*                                                                           */
/* Proxy for a 'Real3x3'.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3X3PROXY_H
#define ARCANE_UTILS_REAL3X3PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Real3Proxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * \brief Proxy for a 'Real3x3'.
*/
class ARCANE_UTILS_EXPORT Real3x3Proxy
{
 public:
 
  //! Constructs the triplet (ax,ay,az)
  Real3x3Proxy(Real3x3& value,const MemoryAccessInfo& info)
  : x(value.x,info), y(value.y,info), z(value.z,info), m_value(value), m_info(info) {}

  //! Constructs a triplet identical to \a f
  Real3x3Proxy(const Real3x3Proxy& f)
  : x(f.x), y(f.y), z(f.z), m_value(f.m_value), m_info(f.m_info) {}
  const Real3x3& operator=(const Real3x3Proxy f)
    { x=f.x; y=f.y; z=f.z; return m_value; }
  const Real3x3& operator=(Real3x3 f)
    { x=f.x; y=f.y; z=f.z; return m_value; }

  //! Assigns the triplet (v,v,v) to the instance.
  const Real3x3& operator= (Real v)
    { x = y = z = v; return m_value; }
  operator const Real3x3&() const
    {
      return getValue();
    }
  //operator Real3x3()
  //{
  //  return getValueMutable();
  //}

 public:

  Real3Proxy x; //!< first element of the triplet
  Real3Proxy y; //!< first element of the triplet
  Real3Proxy z; //!< first element of the triplet

 private:

  Real3x3& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Returns a copy of the triplet.
  Real3x3 copy() const  { return m_value; }

  //! Resets the triplet using default constructors.
  Real3x3Proxy& reset() { x.reset(); y.reset(); z.reset(); return (*this); }

  //! Assigns the triplet (ax,ay,az) to the instance
  Real3x3Proxy& assign(Real3 ax,Real3 ay,Real3 az)
    { x = ax; y = ay; z = az; return (*this); }

  //! Copies the triplet \a f
  Real3x3Proxy& assign(Real3x3 f)
    { x = f.x; y = f.y; z = f.z; return (*this); }

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
  bool isNearlyZero() const
    { return x.isNearlyZero() && y.isNearlyZero() && z.isNearlyZero(); }

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
  Real3x3Proxy& add(Real3x3 b) { x += b.x; y += b.y; z += b.z; return (*this); }

  //! Subtracts \a b from the triplet
  Real3x3Proxy& sub(Real3x3 b) { x -= b.x; y -= b.y; z -= b.z; return (*this); }

  //! Adds \a b to each component of the triplet
  Real3x3Proxy& addSame(Real3 b) { x += b; y += b; z += b; return (*this); }

  //! Subtracts \a b from each component of the triplet
  Real3x3Proxy& subSame(Real3 b) { x -= b; y -= b; z -= b; return (*this); }

  //! Multiplies each component of the triplet by \a b
  Real3x3Proxy& mulSame(Real3 b) { x *= b; y *= b; z *= b; return (*this); }

  //! Divides each component of the triplet by \a b
  Real3x3Proxy& divSame(Real3 b) { x /= b; y /= b; z /= b; return (*this); }

  //! Adds \a b to the triplet.
  Real3x3Proxy& operator+= (Real3x3 b) { return add(b); }

  //! Subtracts \a b from the triplet
  Real3x3Proxy& operator-= (Real3x3 b) { return sub(b); }

  //! Multiplies each component of the matrix by the real \a b
  void operator*= (Real b)  { x *= b; y *= b; z *= b; }

  //! Divides each component of the matrix by the real \a b
  void operator/= (Real  b) { x /= b; y /= b; z /= b; }

  //! Creates a triplet that equals this triplet added to \a b
  Real3x3 operator+(Real3x3 b)  const { return Real3x3(x+b.x,y+b.y,z+b.z); }

  //! Creates a triplet that equals \a b subtracted from this triplet
  Real3x3 operator-(Real3x3 b)  const { return Real3x3(x-b.x,y-b.y,z-b.z); }

  //! Creates a tensor opposite to the current tensor
  Real3x3 operator-() const { return Real3x3(-x,-y,-z); }

 public:
  const Real3x3& getValue() const
    {
      m_info.setRead();
      return m_value;
    }
  Real3x3& getValueMutable()
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
  static bool _eq(Real a,Real b)
    { return TypeEqualT<Real>::isEqual(a,b); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the triplet \a t to the stream \a o
 * \relates Real3
 */
inline std::ostream&
operator<< (std::ostream& o,Real3x3Proxy t)
{
  return t.printXyz(o);
}

/*!
 * \brief Reads the triplet \a t from the stream \a o.
 * \relates Real3
 */
inline std::istream&
operator>> (std::istream& i,Real3x3Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(const Real3x3& a,const Real3x3Proxy& b)
{
  return a==b.getValue();
}
inline bool
operator==(const Real3x3Proxy& a,const Real3x3& b)
{
  return a.getValue()==b;
}
inline bool
operator==(const Real3x3Proxy& a,const Real3x3Proxy& b)
{
  return a.getValue()==b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator!=(const Real3x3& a,const Real3x3Proxy& b)
{
  return a!=b.getValue();
}
inline bool
operator!=(const Real3x3Proxy& a,const Real3x3& b)
{
  return a.getValue()!=b;
}
inline bool
operator!=(const Real3x3Proxy& a,const Real3x3Proxy& b)
{
  return a.getValue()!=b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Multiplication by a scalar. */
inline Real3x3
operator*(Real sca,Real3x3Proxy vec)
{
  return Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Multiplication by a scalar. */
inline Real3x3
operator*(const Real3x3Proxy& vec,Real sca)
{
  return Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Division by a scalar. */
inline Real3x3
operator/(const Real3x3Proxy& vec,Real sca)
{
  return Real3x3(vec.x/sca,vec.y/sca,vec.z/sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Comparison operator.
 *
 * This operator allows sorting Real3s for example
 * when using std::set
 */
inline bool
operator<(Real3x3Proxy v1,Real3x3Proxy v2)
{
  if (v1.x==v2.x){
    if (v1.y==v2.y)
      return v1.z<v2.z;
    else
      return v1.y<v2.y;
  }
  return (v1.x<v2.x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
