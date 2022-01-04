// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2x2Proxy.h                                              (C) 2000-2008 */
/*                                                                           */
/* Proxy du type 'Real2x2'.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2X2PROXY_H
#define ARCANE_UTILS_REAL2X2PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real2Proxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Proxy du type 'Real2x2'.
 */
class ARCANE_UTILS_EXPORT Real2x2Proxy
{
 public:
 
  //! Construit le couple (ax,ay)
  Real2x2Proxy(Real2x2& value,const MemoryAccessInfo& info) 
  : x(value.x,info), y(value.y,info), m_value(value), m_info(info) {}
  //! Construit un couple identique à \a f
  Real2x2Proxy(const Real2x2Proxy& f)
  : x(f.x), y(f.y), m_value(f.m_value), m_info(f.m_info) {}
  const Real2x2Proxy& operator=(const Real2x2Proxy& f)
    { x=f.x; y=f.y; return (*this); }
  const Real2x2Proxy& operator=(Real2x2 f)
    { x=f.x; y=f.y; return (*this); }
  //! Affecte à l'instance le couple (v,v,v).
  const Real2x2Proxy& operator=(Real v)
    { x = v; y = v; return (*this); }

 public:

  Real2Proxy x; //!< Première composante
  Real2Proxy y; //!< Deuxième composante

 private:

  Real2x2& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Retourne une copie du couple.
  Real2x2 copy() const { return getValue(); }
  //! Réinitialise le couple avec les constructeurs par défaut.
  Real2x2Proxy& reset() { *this = Real2x2::null(); return (*this); }
  //! Affecte à l'instance le couple (ax,ay,az)
  Real2x2Proxy& assign(Real2 ax,Real2 ay)
    { x = ax; y = ay; return (*this); }
  //! Copie le couple \a f
  Real2x2Proxy& assign(Real2x2Proxy f)
    { x = f.x; y = f.y; return (*this); }
  operator const Real2x2&() const
    {
      return getValue();
    }
  //operator Real2x2&()
  //{
  //  return getValueMutable();
  //}
  /*!
   * \brief Compare la matrice avec la matrice nulle.
   *
   * La matrice est nulle si et seulement si chacune de ses composant
   * est inférieure à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon\f]
   *
   * \retval true si la matrice est égale à la matrice nulle,
   * \retval false sinon.
   */
  bool isNearlyZero() const
    { return x.isNearlyZero() && y.isNearlyZero(); }
  /*!
   * \brief Lit la matrice sur le flot \a i
   * La matrice est lue sous la forme de trois Real2.
   */
  istream& assign(istream& i);
  //! Ecrit le couple sur le flot \a o lisible par un assign()
  ostream& print(ostream& o) const;
  //! Ecrit le couple sur le flot \a o sous la forme (x,y,z)
  ostream& printXy(ostream& o) const;

  //! Ajoute \a b au couple
  Real2x2Proxy& add(Real2x2 b) { x+=b.x; y+=b.y; return (*this); }
  //! Soustrait \a b au couple
  Real2x2Proxy& sub(Real2x2 b) { x-=b.x; y-=b.y; return (*this); }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  //Real2x2Proxy& mul(Real2x2Proxy b) { x*=b.x; y*=b.y; return (*this); }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  Real2x2Proxy& div(Real2x2 b) { x/=b.x; y/=b.y; return (*this); }
  //! Ajoute \a b à chaque composante du couple
  Real2x2Proxy& addSame(Real2 b) { x+=b; y+=b; return (*this); }
  //! Soustrait \a b à chaque composante du couple
  Real2x2Proxy& subSame(Real2 b) { x-=b; y-=b; return (*this); }
  //! Multiplie chaque composante du couple par \a b
  Real2x2Proxy& mulSame(Real2 b) { x*=b; y*=b; return (*this); }
  //! Divise chaque composante du couple par \a b
  Real2x2Proxy& divSame(Real2 b) { x/=b; y/=b; return (*this); }
  //! Ajoute \a b au couple.
  Real2x2Proxy& operator+=(Real2x2 b) { return add(b); }
  //! Soustrait \a b au couple
  Real2x2Proxy& operator-=(Real2x2 b) { return sub(b); }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  //Real2x2Proxy& operator*=(Real2x2Proxy b) { return mul(b); }
  //! Multiple chaque composante de la matrice par le réel \a b
  void operator*=(Real b) { x*=b; y*=b; }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  //Real2x2Proxy& operator/= (Real2x2 b) { return div(b); }
  //! Divise chaque composante de la matrice par le réel \a b
  void operator/= (Real b) { x/=b; y/=b; }
  //! Créé un couple qui vaut ce couple ajouté à \a b
  Real2x2 operator+(Real2x2 b) const { return Real2x2(x+b.x,y+b.y); }
  //! Créé un couple qui vaut \a b soustrait de ce couple
  Real2x2 operator-(Real2x2 b)  const { return Real2x2(x-b.x,y-b.y); }
  //! Créé un tenseur opposé au tenseur actuel
  Real2x2 operator-() const { return Real2x2(-x,-y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  //Real2x2 operator*(Real2x2Proxy b) const { return Real2x2Proxy(x*b.x,y*b.y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été divisée
   * par la composante correspondante de \a b.
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
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  static bool _eq(Real a,Real b)
    { return TypeEqualT<Real>::isEqual(a,b); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(const Real2x2& a,const Real2x2Proxy& b)
{
  return a==b.getValue();
}
inline bool
operator==(const Real2x2Proxy& a,const Real2x2& b)
{
  return a.getValue()==b;
}
inline bool
operator==(const Real2x2Proxy& a,const Real2x2Proxy& b)
{
  return a.getValue()==b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator!=(const Real2x2& a,const Real2x2Proxy& b)
{
  return a!=b.getValue();
}
inline bool
operator!=(const Real2x2Proxy& a,const Real2x2& b)
{
  return a.getValue()!=b;
}
inline bool
operator!=(const Real2x2Proxy& a,const Real2x2Proxy& b)
{
  return a.getValue()!=b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le couple \a t sur le flot \a o
 * \relates Real2x2Proxy
 */
inline ostream&
operator<< (ostream& o,const Real2x2Proxy& t)
{
  return t.printXy(o);
}
/*!
 * \brief Lit le couple \a t à partir du flot \a o.
 * \relates Real2x2Proxy
 */
inline istream&
operator>> (istream& i,Real2x2Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Multiplication par un scalaire. */
inline Real2x2
operator*(Real sca,const Real2x2Proxy& vec)
{
  return Real2x2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Multiplication par un scalaire. */
inline Real2x2
operator*(const Real2x2Proxy& vec,Real sca)
{
  return Real2x2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Division par un scalaire. */
inline Real2x2
operator/(const Real2x2Proxy& vec,Real sca)
{
  return Real2x2(vec.x/sca,vec.y/sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérateur de comparaison.
 *
 * Cet opérateur permet de trier les Real2 pour les utiliser par exemple
 * dans les std::set
 */
inline bool
operator<(const Real2x2Proxy& v1,const Real2x2Proxy& v2)
{
  return (v1.getValue()<v2.getValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
