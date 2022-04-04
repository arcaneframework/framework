// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2Proxy.h                                                (C) 2000-2008 */
/*                                                                           */
/* Proxy d'un 'Real2'.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2PROXY_H
#define ARCANE_UTILS_REAL2PROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/BuiltInProxy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un vecteur de réel de dimension 2.

 Le vecteur comprend deuxs composantes \a x et \a y qui sont du
 type \b Real.

 \code
 Real2Proxy value (1.0,2.3); // Créé un couple (x=1.0, y=2.3)
 cout << value.x;   // Imprime la composante x 
 value.y += 3.2; // Ajoute 3.2 à la composante \b y
 \endcode
 */
class ARCANE_UTILS_EXPORT Real2Proxy
{
 public:
 
  //! Construit le couplet (ax,ay)
  Real2Proxy(Real2& value,const MemoryAccessInfo& info)
  : x(value.x,info), y(value.y,info), m_value(value), m_info(info) {}
  //! Construit un couple identique à \a f
  Real2Proxy(const Real2Proxy& f)
  : x(f.x), y(f.y), m_value(f.m_value), m_info(f.m_info) {}
  const Real2& operator=(Real2Proxy f)
    { x=f.x; y=f.y; return m_value; }
  const Real2& operator=(Real2 f)
    { x=f.x; y=f.y; return m_value; }
  //! Affecte à l'instance le couple (v,v).
  const Real2& operator=(Real v)
    { x = y = v; return m_value; }
  operator Real2() const
    {
      return getValue();
    }
  //operator Real2&()
  //{
  //  return getValueMutable();
  //}
 public:

  RealProxy x; //!< première composante du couple
  RealProxy y; //!< deuxième composante du couple

 private:

  Real2& m_value;
  MemoryAccessInfo m_info;

 public:

  //! Retourne une copie du couple.
  Real2 copy() const { return m_value; }
  //! Réinitialise le couple avec les constructeurs par défaut.
  Real2Proxy& reset() { x = y = 0.; return (*this); }
  //! Affecte à l'instance le couple (ax,ay,az)
  Real2Proxy& assign(Real ax,Real ay)
    { x = ax; y = ay; return (*this); }
  //! Copie le couple \a f
  Real2Proxy& assign(Real2 f)
    { x = f.x; y = f.y; return (*this); }
  /*!
   * \brief Compare le couple avec le couple nul.
   *
   * Dans le cas d'un type #value_type de type intégral, le couple est
   * nul si et seulement si chacune de ses composantes est égal à 0.
   *
   * Pour #value_type du type flottant (float, double ou #Real), le couple
   * est nul si et seulement si chacune de ses composant est inférieure
   * à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon \f]
   *
   * \retval true si le couple est égal au couple nul,
   * \retval false sinon.
   */
  bool isNearlyZero() const
    {
      return math::isNearlyZero(x.getValue()) && math::isNearlyZero(y.getValue());
    }
  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  Real abs2() const
    { return x*x + y*y; }
  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  Real abs() const
    { return _sqrt(abs2()); }

  /*!
   * \brief Lit un couple sur le flot \a i
   * Le couple est lu sous la forme de trois valeur de type #value_type.
   */
  istream& assign(istream& i);
  //! Ecrit le couple sur le flot \a o lisible par un assign()
  ostream& print(ostream& o) const;
  //! Ecrit le couple sur le flot \a o sous la forme (x,y)
  ostream& printXy(ostream& o) const;

  //! Ajoute \a b au couple
  Real2Proxy& add(Real2 b) { x+=b.x; y+=b.y; return (*this); }
  //! Soustrait \a b au couple
  Real2Proxy& sub(Real2 b) { x-=b.x; y-=b.y; return (*this); }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  Real2Proxy& mul(Real2 b) { x*=b.x; y*=b.y; return (*this); }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  Real2Proxy& div(Real2 b) { x/=b.x; y/=b.y; return (*this); }
  //! Ajoute \a b à chaque composante du couple
  Real2Proxy& addSame(Real b) { x+=b; y+=b; return (*this); }
  //! Soustrait \a b à chaque composante du couple
  Real2Proxy& subSame(Real b) { x-=b; y-=b; return (*this); }
  //! Multiplie chaque composante du couple par \a b
  Real2Proxy& mulSame(Real b) { x*=b; y*=b; return (*this); }
  //! Divise chaque composante du couple par \a b
  Real2Proxy& divSame(Real b) { x/=b; y/=b; return (*this); }
  //! Ajoute \a b au couple.
  Real2Proxy& operator+=(Real2 b) { return add(b); }
  //! Soustrait \a b au couple
  Real2Proxy& operator-=(Real2 b) { return sub(b); }
  //! Multiplie chaque composante du couple par la composant correspondant de \a b
  Real2Proxy& operator*=(Real2 b) { return mul(b); }
  //! Multiplie chaque composante du couple par le réel \a b
  void  operator*=(Real b) { x*=b; y*=b; }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  Real2Proxy& operator/=(Real2 b) { return div(b); }
  //! Divise chaque composante du couple par le réel \a b
  void  operator/=(Real  b) { x/=b; y/=b; }
  //! Créé un couple qui vaut ce couple ajouté à \a b
  Real2 operator+(Real2 b)  const { return Real2(x+b.x,y+b.y); }
  //! Créé un couple qui vaut \a b soustrait de ce couple
  Real2 operator-(Real2 b)  const { return Real2(x-b.x,y-b.y); }
  //! Créé un couple opposé au couple actuel
  Real2 operator-() const { return Real2(-x,-y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  Real2 operator*(Real2 b) const { return Real2(x*b.x,y*b.y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été divisée
   * par la composante correspondante de \a b.
   */
  Real2 operator/(Real2 b) const { return Real2(x/b.x,y/b.y); }
										
  /*!
   * \brief Normalise le couple.
   * 
   * Si le couple est non nul, divise chaque composante par la norme du couple
   * (abs()), de telle sorte qu'après l'appel à cette méthode, abs() valent \a 1.
   * Si le couple est nul, ne fait rien.
   */
  Real2Proxy& normalize()
    {
      Real d = abs();
      if (!math::isZero(d))
        divSame(d);
      return (*this);
    }
	
  /*!
   * \brief Compare le couple à \a b.
   *
   * Dans le cas d'un type #value_type de type intégral, deux couples
   * sont égaux si et seulement si chacune de leur composant sont strictement
   * égales.
   *
   * Pour #value_type du type flottant (float, double ou #Real), deux couples
   * sont identiques si et seulement si la valeur absolue de la différence
   * entre chacune de leur composant correspondante est inférieure
   * à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=B \Leftrightarrow |A.x-B.x|<\epsilon,|A.y-B.y|<\epsilon,|A.z-B.z|<\epsilon \f]
   * \retval true si les deux couples sont égaux,
   * \retval false sinon.
   */
  bool operator==(Real2 b) const
    { return _eq(x,b.x) && _eq(y,b.y); }
  /*!
   * \brief Compare deux couples.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux couples sont différents,
   * \retval false sinon.
   */
  bool operator!=(Real2 b) const
    { return !operator==(b); }

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
inline Real2 operator*(Real sca,const Real2Proxy& vec)
{
  return Real2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication par un scalaire.
 */
inline Real2
operator*(const Real2Proxy& vec,Real sca)
{
  return Real2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Division par un scalaire.
 */
inline Real2
operator/(const Real2Proxy& vec,Real sca)
{
  return Real2(vec.x/sca,vec.y/sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérateur de comparaison.
 *
 * Cet opérateur permet de trier les Real2Proxy pour les utiliser par exemple
 * dans les std::set
 */
inline bool
operator<(const Real2Proxy& v1,const Real2Proxy& v2)
{
  return v1.getValue() < v2.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le couple \a t sur le flot \a o
 * \relates Real2Proxy
 */
inline ostream&
operator<< (ostream& o,Real2Proxy t)
{
  return t.printXy(o);
}
/*!
 * \brief Lit le couple \a t à partir du flot \a o.
 * \relates Real2Proxy
 */
inline istream&
operator>> (istream& i,Real2Proxy& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

