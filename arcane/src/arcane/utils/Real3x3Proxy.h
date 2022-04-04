// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3x3Proxy.h                                              (C) 2000-2008 */
/*                                                                           */
/* Proxy d'un 'Real3x3'.                                                     */
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
 * \brief Proxy d'un 'Real3x3'.
*/
class ARCANE_UTILS_EXPORT Real3x3Proxy
{
 public:
 
  //! Construit le triplet (ax,ay,az)
  Real3x3Proxy(Real3x3& value,const MemoryAccessInfo& info)
  : x(value.x,info), y(value.y,info), z(value.z,info), m_value(value), m_info(info) {}
  //! Construit un triplet identique à \a f
  Real3x3Proxy(const Real3x3Proxy& f)
  : x(f.x), y(f.y), z(f.z), m_value(f.m_value), m_info(f.m_info) {}
  const Real3x3& operator=(const Real3x3Proxy f)
    { x=f.x; y=f.y; z=f.z; return m_value; }
  const Real3x3& operator=(Real3x3 f)
    { x=f.x; y=f.y; z=f.z; return m_value; }
  //! Affecte à l'instance le triplet (v,v,v).
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

  Real3Proxy x; //!< premier élément du triplet
  Real3Proxy y; //!< premier élément du triplet
  Real3Proxy z; //!< premier élément du triplet

 private:

  Real3x3& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Retourne une copie du triplet.
  Real3x3 copy() const  { return m_value; }
  //! Réinitialise le triplet avec les constructeurs par défaut.
  Real3x3Proxy& reset() { x.reset(); y.reset(); z.reset(); return (*this); }
  //! Affecte à l'instance le triplet (ax,ay,az)
  Real3x3Proxy& assign(Real3 ax,Real3 ay,Real3 az)
    { x = ax; y = ay; z = az; return (*this); }
  //! Copie le triplet \a f
  Real3x3Proxy& assign(Real3x3 f)
    { x = f.x; y = f.y; z = f.z; return (*this); }
  /*!
   * \brief Compare la matrice avec la matrice nulle.
   *
   * La matrice est nulle si et seulement si chacune de ses composant
   * est inférieure à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
   *
   * \retval true si la matrice est égale à la matrice nulle,
   * \retval false sinon.
   */
  bool isNearlyZero() const
    { return x.isNearlyZero() && y.isNearlyZero() && z.isNearlyZero(); }
  /*!
   * \brief Lit la matrice sur le flot \a i
   * La matrice est lue sous la forme de trois Real3.
   */
  std::istream& assign(std::istream& i);
  //! Ecrit le triplet sur le flot \a o lisible par un assign()
  std::ostream& print(std::ostream& o) const;
  //! Ecrit le triplet sur le flot \a o sous la forme (x,y,z)
  std::ostream& printXyz(std::ostream& o) const;

  //! Ajoute \a b au triplet
  Real3x3Proxy& add(Real3x3 b) { x += b.x; y += b.y; z += b.z; return (*this); }
  //! Soustrait \a b au triplet
  Real3x3Proxy& sub(Real3x3 b) { x -= b.x; y -= b.y; z -= b.z; return (*this); }
  //! Ajoute \a b à chaque composante du triplet
  Real3x3Proxy& addSame(Real3 b) { x += b; y += b; z += b; return (*this); }
  //! Soustrait \a b à chaque composante du triplet
  Real3x3Proxy& subSame(Real3 b) { x -= b; y -= b; z -= b; return (*this); }
  //! Multiplie chaque composante du triplet par \a b
  Real3x3Proxy& mulSame(Real3 b) { x *= b; y *= b; z *= b; return (*this); }
  //! Divise chaque composante du triplet par \a b
  Real3x3Proxy& divSame(Real3 b) { x /= b; y /= b; z /= b; return (*this); }
  //! Ajoute \a b au triplet.
  Real3x3Proxy& operator+= (Real3x3 b) { return add(b); }
  //! Soustrait \a b au triplet
  Real3x3Proxy& operator-= (Real3x3 b) { return sub(b); }
  //! Multiple chaque composante de la matrice par le réel \a b
  void operator*= (Real b)  { x *= b; y *= b; z *= b; }
  //! Divise chaque composante de la matrice par le réel \a b
  void operator/= (Real  b) { x /= b; y /= b; z /= b; }
  //! Créé un triplet qui vaut ce triplet ajouté à \a b
  Real3x3 operator+(Real3x3 b)  const { return Real3x3(x+b.x,y+b.y,z+b.z); }
  //! Créé un triplet qui vaut \a b soustrait de ce triplet
  Real3x3 operator-(Real3x3 b)  const { return Real3x3(x-b.x,y-b.y,z-b.z); }
  //! Créé un tenseur opposé au tenseur actuel
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
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  static bool _eq(Real a,Real b)
    { return TypeEqualT<Real>::isEqual(a,b); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le triplet \a t sur le flot \a o
 * \relates Real3
 */
inline std::ostream&
operator<< (std::ostream& o,Real3x3Proxy t)
{
  return t.printXyz(o);
}
/*!
 * \brief Lit le triplet \a t à partir du flot \a o.
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
/*! \brief Multiplication par un scalaire. */
inline Real3x3
operator*(Real sca,Real3x3Proxy vec)
{
  return Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Multiplication par un scalaire. */
inline Real3x3
operator*(const Real3x3Proxy& vec,Real sca)
{
  return Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Division par un scalaire. */
inline Real3x3
operator/(const Real3x3Proxy& vec,Real sca)
{
  return Real3x3(vec.x/sca,vec.y/sca,vec.z/sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérateur de comparaison.
 *
 * Cet opérateur permet de trier les Real3 pour les utiliser par exemple
 * dans les std::set
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
