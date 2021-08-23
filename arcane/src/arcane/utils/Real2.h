// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2.h                                                     (C) 2000-2021 */
/*                                                                           */
/* Vecteur à 2 dimensions de 'Real'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_REAL2_H
#define ARCANE_DATATYPE_REAL2_H
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
  Real x; //!< première composante du couple
  Real y; //!< deuxième composante du couple
  /**
   * Accès en lecture seule à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y  ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  const Real& operator[](const Integer& i) const
  {
    ARCANE_ASSERT(((i>=0)&&(i<2)),("Trying to use an index different than 0 or 1 on a Real2"));
    return (&x)[i];
  }

  /**
   * Accès à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  Real& operator[](const Integer& i)
  {
    ARCANE_ASSERT(((i>=0)&&(i<2)),("Trying to use an index different than 0 or 1 on a Real2"));
    return (&x)[i];
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un vecteur de réel de dimension 2.

 Le vecteur comprend deuxs composantes \a x et \a y qui sont du
 type \b Real.

 \code
 Real2 value (1.0,2.3); // Créé un couple (x=1.0, y=2.3)
 cout << value.x;   // Imprime la composante x 
 value.y += 3.2; // Ajoute 3.2 à la composante \b y
 \endcode
 */
class ARCANE_UTILS_EXPORT Real2
: public Real2POD
{
 public:
 
  //! Construit le vecteur nul.
  ARCCORE_HOST_DEVICE Real2() { x=0.; y=0.; }
  //! Construit le couplet (ax,ay)
  ARCCORE_HOST_DEVICE Real2(Real ax,Real ay) { x=ax; y=ay; }
  //! Construit un couple identique à \a f
  ARCCORE_HOST_DEVICE Real2(const Real2& f)	{ x=f.x; y=f.y; }
  //! Construit un coupe identique à \a f
  ARCCORE_HOST_DEVICE explicit Real2(const Real2POD& f) { x=f.x; y=f.y; }
  ARCCORE_HOST_DEVICE const Real2& operator=(Real2 f)
  { x=f.x; y=f.y; return (*this); }
  //! Affecte à l'instance le couple (v,v).
  ARCCORE_HOST_DEVICE const Real2&  operator= (Real v)
  { x = y = v; return (*this); }

 public:

  ARCCORE_HOST_DEVICE static Real2 null() { return Real2(0.,0.); }

 public:

  //! Retourne une copie du couple.
  ARCCORE_HOST_DEVICE Real2 copy() const { return (*this); }
  //! Réinitialise le couple avec les constructeurs par défaut.
  ARCCORE_HOST_DEVICE Real2& reset() { x = y = 0.; return (*this); }
  //! Affecte à l'instance le couple (ax,ay,az)
  ARCCORE_HOST_DEVICE Real2& assign(Real ax,Real ay) { x = ax; y = ay; return (*this); }
  //! Copie le couple \a f
  ARCCORE_HOST_DEVICE Real2& assign(Real2 f) { x = f.x; y = f.y; return (*this); }
  /*!
   * \brief Indique si l'instance est proche de l'instance nulle.
   *
   * \retval true si math::isNearlyZero() est vrai pour chaque composante.
   * \retval false sinon.
   */
  ARCCORE_HOST_DEVICE bool isNearlyZero() const
  {
    return math::isNearlyZero(x) && math::isNearlyZero(y);
  }

  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  ARCCORE_HOST_DEVICE Real squareNormL2() const { return x*x + y*y; }
  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_HOST_DEVICE Real normL2() const { return _sqrt(squareNormL2()); }

  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  ARCCORE_DEPRECATED_2021("Use squareNormL2() instead")
  ARCCORE_HOST_DEVICE Real abs2() const { return x*x + y*y; }
  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_DEPRECATED_2021("Use normL2() instead")
  ARCCORE_HOST_DEVICE Real abs() const { return _sqrt(squareNormL2()); }
  //! Valeur absolue composante par composante.
  ARCCORE_HOST_DEVICE Real2 absolute() const { return Real2(math::abs(x),math::abs(y)); }

  /*!
   * \brief Lit un couple sur le flot \a i
   * Le couple est lu sous la forme de trois valeur de type #value_type.
   */
  std::istream& assign(std::istream& i);
  //! Ecrit le couple sur le flot \a o lisible par un assign()
  std::ostream& print(std::ostream& o) const;
  //! Ecrit le couple sur le flot \a o sous la forme (x,y)
  std::ostream& printXy(std::ostream& o) const;

  //! Ajoute \a b au couple
  ARCCORE_HOST_DEVICE Real2& add(Real2 b) { x+=b.x; y+=b.y; return (*this); }
  //! Soustrait \a b au couple
  ARCCORE_HOST_DEVICE Real2& sub(Real2 b) { x-=b.x; y-=b.y; return (*this); }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  ARCCORE_HOST_DEVICE Real2& mul(Real2 b) { x*=b.x; y*=b.y; return (*this); }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  ARCCORE_HOST_DEVICE Real2& div(Real2 b) { x/=b.x; y/=b.y; return (*this); }
  //! Ajoute \a b à chaque composante du couple
  ARCCORE_HOST_DEVICE Real2& addSame(Real b) { x+=b; y+=b; return (*this); }
  //! Soustrait \a b à chaque composante du couple
  ARCCORE_HOST_DEVICE Real2& subSame(Real b) { x-=b; y-=b; return (*this); }
  //! Multiplie chaque composante du couple par \a b
  ARCCORE_HOST_DEVICE Real2& mulSame(Real b) { x*=b; y*=b; return (*this); }
  //! Divise chaque composante du couple par \a b
  ARCCORE_HOST_DEVICE Real2& divSame(Real b) { x/=b; y/=b; return (*this); }
  //! Ajoute \a b au couple.
  ARCCORE_HOST_DEVICE Real2& operator+=(Real2 b) { return add(b); }
  //! Soustrait \a b au couple
  ARCCORE_HOST_DEVICE Real2& operator-=(Real2 b) { return sub(b); }
  //! Multiplie chaque composante du couple par la composant correspondant de \a b
  ARCCORE_HOST_DEVICE Real2& operator*=(Real2 b) { return mul(b); }
  //! Multiplie chaque composante du couple par le réel \a b
  ARCCORE_HOST_DEVICE void  operator*=(Real  b) { x*=b; y*=b; }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  ARCCORE_HOST_DEVICE Real2& operator/=(Real2 b) { return div(b); }
  //! Divise chaque composante du couple par le réel \a b
  ARCCORE_HOST_DEVICE void  operator/=(Real  b) { x/=b; y/=b; }
  //! Créé un couple qui vaut ce couple ajouté à \a b
  ARCCORE_HOST_DEVICE Real2 operator+(Real2 b)  const { return Real2(x+b.x,y+b.y); }
  //! Créé un couple qui vaut \a b soustrait de ce couple
  ARCCORE_HOST_DEVICE Real2 operator-(Real2 b)  const { return Real2(x-b.x,y-b.y); }
  //! Créé un couple opposé au couple actuel
  ARCCORE_HOST_DEVICE Real2 operator-() const { return Real2(-x,-y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  ARCCORE_HOST_DEVICE Real2 operator*(Real2 b) const { return Real2(x*b.x,y*b.y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été divisée
   * par la composante correspondante de \a b.
   */
  ARCCORE_HOST_DEVICE Real2 operator/(Real2 b) const { return Real2(x/b.x,y/b.y); }
										
  /*!
   * \brief Normalise le couple.
   * 
   * Si le couple est non nul, divise chaque composante par la norme du couple
   * (abs()), de telle sorte qu'après l'appel à cette méthode, abs() valent \a 1.
   * Si le couple est nul, ne fait rien.
   */
  Real2& normalize()
  {
    Real d = normL2();
    if (!math::isZero(d))
      divSame(d);
    return (*this);
  }
	
  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y.
   * \retval false sinon.
   */
  ARCCORE_HOST_DEVICE bool operator==(Real2 b) const
  {
    return _eq(x,b.x) && _eq(y,b.y);
  }

  /*!
   * \brief Compare deux couples.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux couples sont différents,
   * \retval false sinon.
   */
  ARCCORE_HOST_DEVICE bool operator!=(Real2 b) const { return !operator==(b); }


  /** 
    * Accès à la @a i eme composante du Real3POD
    * @note ne fonctionne que pour x, y et z ordonnées dans le POD
    * 
    * @param i numéro de la composante à retourner
    * 
    * @return (&x)[i]
    */
  Real operator[](Integer i) const
  {
    ARCANE_ASSERT(((i>=0)&&(i<2)),("Trying to use an index different than 0 or 1 on a Real2"));
    return (&x)[i];
  }
  
  Real& operator[](Integer i)
  {
    ARCANE_ASSERT(((i>=0)&&(i<2)),("Trying to use an index different than 0 or 1 on a Real2"));
    return (&x)[i];
  }

 private:

  /*!
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  ARCCORE_HOST_DEVICE static bool _eq(Real a,Real b) { return math::isEqual(a,b); }
  //! Retourne la racine carrée de \a a
  ARCCORE_HOST_DEVICE static Real _sqrt(Real a) { return math::sqrt(a); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication par un scalaire.
 */
inline Real2 operator*(Real sca,Real2 vec)
{
  return Real2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication par un scalaire.
 */
inline Real2
operator*(Real2 vec,Real sca)
{
  return Real2(vec.x*sca,vec.y*sca);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Division par un scalaire.
 */
inline Real2
operator/(Real2 vec,Real sca)
{
  return Real2(vec.x/sca,vec.y/sca);
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
operator<(Real2 v1,Real2 v2)
{
  if (v1.x==v2.x){
    return v1.y<v2.y;
  }
  return (v1.x<v2.x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le couple \a t sur le flot \a o
 * \relates Real2
 */
inline std::ostream&
operator<< (std::ostream& o,Real2 t)
{
  return t.printXy(o);
}
/*!
 * \brief Lit le couple \a t à partir du flot \a o.
 * \relates Real2
 */
inline std::istream&
operator>> (std::istream& i,Real2& t)
{
  return t.assign(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

