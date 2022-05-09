// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3Proxy.h                                                (C) 2000-2022 */
/*                                                                           */
/* Proxy d'un 'Real3'.                                                       */
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
 * \brief Proxy d'un Real3.
 */
class ARCANE_UTILS_EXPORT Real3Proxy
: public Real3POD
{
 public:
 
  //! Construit le triplet (ax,ay,az)
  Real3Proxy(Real3& value,const MemoryAccessInfo& info)
  : x(value.x,info), y(value.y,info), z(value.z,info), m_value(value), m_info(info) {}
  //! Construit un triplet identique à \a f
  Real3Proxy(const Real3Proxy& f)
	: x(f.x), y(f.y), z(f.z), m_value(f.m_value), m_info(f.m_info) {}
  Real3 operator=(const Real3Proxy f)
    { x = f.x; y = f.y; z = f.z; return m_value; }
  Real3 operator=(Real3 f)
    { x = f.x; y = f.y; z = f.z; return m_value; }
  //! Affecte à l'instance le triplet (v,v,v).
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

  RealProxy x; //!< première composante du triplet
  RealProxy y; //!< deuxième composante du triplet
  RealProxy z; //!< troisième composante du triplet

 private:

  Real3& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Retourne une copie du triplet.
  Real3 copy() const { return Real3(x,y,z); }
  //! Réinitialise le triplet avec les constructeurs par défaut.
  Real3Proxy& reset() { x = y = z = 0.; return (*this); }
  //! Affecte à l'instance le triplet (ax,ay,az)
  Real3Proxy& assign(Real ax,Real ay,Real az)
    { x = ax; y = ay; z = az; return (*this); }
  //! Copie le triplet \a f
  Real3Proxy& assign(Real3 f)
    { x = f.x; y = f.y; z = f.z; return (*this); }
  /*!
   * \brief Compare le triplet avec le triplet nul.
   *
   * Dans le cas d'un type #value_type de type intégral, le triplet est
   * nul si et seulement si chacune de ses composantes est égal à 0.
   *
   * Pour #value_type du type flottant (float, double ou #Real), le triplet
   * est nul si et seulement si chacune de ses composant est inférieure
   * à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
   *
   * \retval true si le triplet est égal au triplet nul,
   * \retval false sinon.
   */
  bool isNearlyZero() const
    {
      return math::isNearlyZero(x.getValue()) && math::isNearlyZero(y.getValue()) && math::isNearlyZero(z.getValue());
    }
  //! Retourne la norme au carré du triplet \f$x^2+y^2+z^2\f$
  Real abs2() const
    { return x*x + y*y + z*z; }
  //! Retourne la norme du triplet \f$\sqrt{x^2+y^2+z^2}\f$
  Real abs() const
    { return _sqrt(abs2()); }

  /*!
   * \brief Lit un triplet sur le flot \a i
   * Le triplet est lu sous la forme de trois valeur de type #value_type.
   */
  std::istream& assign(std::istream& i);
  //! Ecrit le triplet sur le flot \a o lisible par un assign()
  std::ostream& print(std::ostream& o) const;
  //! Ecrit le triplet sur le flot \a o sous la forme (x,y,z)
  std::ostream& printXyz(std::ostream& o) const;

  //! Ajoute \a b au triplet
  Real3Proxy& add(Real3 b) { x+=b.x; y+=b.y; z+=b.z; return (*this); }
  //! Soustrait \a b au triplet
  Real3Proxy& sub(Real3 b) { x-=b.x; y-=b.y; z-=b.z; return (*this); }
  //! Multiple chaque composante du triplet par la composant correspondant de \a b
  Real3Proxy& mul(Real3 b) { x*=b.x; y*=b.y; z*=b.z; return (*this); }
  //! Divise chaque composante du triplet par la composant correspondant de \a b
  Real3Proxy& div(Real3 b) { x/=b.x; y/=b.y; z/=b.z; return (*this); }
  //! Ajoute \a b à chaque composante du triplet
  Real3Proxy& addSame(Real b) { x+=b; y+=b; z+=b; return (*this); }
  //! Soustrait \a b à chaque composante du triplet
  Real3Proxy& subSame(Real b) { x-=b; y-=b; z-=b; return (*this); }
  //! Multiplie chaque composante du triplet par \a b
  Real3Proxy& mulSame(Real b) { x*=b; y*=b; z*=b; return (*this); }
  //! Divise chaque composante du triplet par \a b
  Real3Proxy& divSame(Real b) { x/=b; y/=b; z/=b; return (*this); }

    //! Créé un triplet qui vaut le produit vectoriel de ce triplet avec le triplet \a b.
  Real3 cross(Real3 b)  const { return Real3(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
  //! Retourne le produit scalaire de ce triplet avec le triplet \a b.
  Real dot(Real3 b) const { return x*b.x + y*b.y + z*b.z; }

  //! Ajoute \a b au triplet.
  Real3Proxy& operator+= (Real3 b) { return add(b); }
  //! Soustrait \a b au triplet
  Real3Proxy& operator-= (Real3 b) { return sub(b); }
  //! Multiple chaque composante du triplet par la composant correspondant de \a b
  Real3Proxy& operator*= (Real3 b) { return mul(b); }
  //! Multiple chaque composante du triplet par le réel \a b
  void operator*= (Real  b) { x*=b; y*=b; z*=b; }
  //! Divise chaque composante du triplet par la composant correspondant de \a b
  Real3Proxy& operator/= (Real3 b) { return div(b); }
  //! Divise chaque composante du triplet par le réel \a b
  void operator/= (Real  b) { x/=b; y/=b; z/=b; }
  //! Créé un triplet qui vaut ce triplet ajouté à \a b
  Real3 operator+  (Real3 b)  const { return Real3(x+b.x,y+b.y,z+b.z); }
  //! Créé un triplet qui vaut \a b soustrait de ce triplet
  Real3 operator-  (Real3 b)  const { return Real3(x-b.x,y-b.y,z-b.z); }
  //! Créé un triplet opposé au triplet actuel
  Real3 operator-() const { return Real3(-x,-y,-z); }
  /*!
   * \brief Créé un triplet qui vaut ce triplet dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  Real3 operator*(Real3 b) const { return Real3(x*b.x,y*b.y,z*b.z); }
  /*!
   * \brief Créé un triplet qui vaut ce triplet dont chaque composant a été divisée
   * par la composante correspondante de \a b.
   */
  Real3 operator/(Real3 b) const { return Real3(x/b.x,y/b.y,z/b.z); }
										
  /*!
   * \brief Normalise le triplet.
   * 
   * Si le triplet est non nul, divise chaque composante par la norme du triplet
   * (abs()), de telle sorte qu'après l'appel à cette méthode, abs() valent \a 1.
   * Si le triplet est nul, ne fait rien.
   */
  Real3Proxy& normalize()
    {
      Real d = abs();
      if (!math::isZero(d))
        divSame(d);
      return (*this);
    }
	
  /*!
   * \brief Compare le triplet à \a b.
   *
   * Dans le cas d'un type #value_type de type intégral, deux triplets
   * sont égaux si et seulement si chacune de leur composant sont strictement
   * égales.
   *
   * Pour #value_type du type flottant (float, double ou #Real), deux triplets
   * sont identiques si et seulement si la valeur absolue de la différence
   * entre chacune de leur composant correspondante est inférieure
   * à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=B \Leftrightarrow |A.x-B.x|<\epsilon,|A.y-B.y|<\epsilon,|A.z-B.z|<\epsilon \f]
   * \retval true si les deux triplets sont égaux,
   * \retval false sinon.
   */
  friend bool operator==(Real3Proxy a,Real3Proxy b)
  { return  _eq(a.x,b.x) &&  _eq(a.y,b.y) && _eq(a.z,b.z); }
  /*!
   * \brief Compare deux triplets.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux triplets sont différents,
   * \retval false sinon.
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
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  static bool _eq(Real a,Real b)
    { return math::isEqual(a,b); }
  //! Retourne la racine carrée de \a a
  static Real _sqrt(Real a)
    { return math::sqrt(a); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication par un scalaire.
 */
inline Real3 operator*(Real sca,Real3Proxy vec)
{
  return Real3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication par un scalaire.
 */
inline Real3
operator*(Real3Proxy vec,Real sca)
{
  return Real3(vec.x*sca,vec.y*sca,vec.z*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Division par un scalaire.
 */
inline Real3
operator/(Real3Proxy vec,Real sca)
{
  return Real3(vec.x/sca,vec.y/sca,vec.z/sca);
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
operator<(const Real3Proxy v1,const Real3Proxy v2)
{
  return v1.getValue()<v2.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le triplet \a t sur le flot \a o
 * \relates Real3
 */
inline std::ostream&
operator<< (std::ostream& o,Real3Proxy t)
{
  return t.printXyz(o);
}
/*!
 * \brief Lit le triplet \a t à partir du flot \a o.
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
