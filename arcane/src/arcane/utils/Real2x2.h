// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2x2.h                                                   (C) 2000-2024 */
/*                                                                           */
/* Matrice 2x2 de 'Real'.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2X2_H
#define ARCANE_UTILS_REAL2X2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure POD pour un Real2x2.
 */
struct Real2x2POD
{
 public:

  Real2POD x;
  Real2POD y;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant une matrice de réel de dimension 2x2.
 *
 * La matrice comprend deux composantes \a x et \a y qui sont du
 * type \b Real2. Par exemple:
 *
 * \code
 * Real2x2 matrix;
 * matrix.x.y = 2.;
 * matrix.y.y = 3.;
 * \endcode
 */
class ARCANE_UTILS_EXPORT Real2x2
{
 public:

  //! Construit la matrice nulle
  constexpr ARCCORE_HOST_DEVICE Real2x2()
  : x(Real2::null())
  , y(Real2::null())
  {}
  //! Construit le couple (ax,ay)
  constexpr ARCCORE_HOST_DEVICE Real2x2(Real2 ax, Real2 ay)
  : x(ax)
  , y(ay)
  {}
  /*!
   * \brief Construit le couple ((ax,bx),(ay,by)).
   * \deprecated Utiliser Real2x2(Real2 a,Real2 b) à la place.
   */
  ARCANE_DEPRECATED_116 Real2x2(Real ax, Real ay, Real bx, Real by)
  : x(ax, bx)
  , y(ay, by)
  {}
  //! Construit un couple identique à \a f
  Real2x2(const Real2x2& f) = default;
  //! Construit un couple identique à \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real2x2(const Real2x2POD& f)
  : x(f.x)
  , y(f.y)
  {}
  //! Construit l'instance avec le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real2x2(Real v)
  {
    x = y = v;
  }
  //! Opérateur de recopie
  Real2x2& operator=(const Real2x2& f) = default;

  //! Affecte à l'instance le couple (v,v,v).
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator=(Real v)
  {
    x = y = v;
    return (*this);
  }

 public:

  Real2 x; //!< Première composante
  Real2 y; //!< Deuxième composante

 public:

  //! Construit la matrice nulle
  constexpr ARCCORE_HOST_DEVICE static Real2x2 null() { return Real2x2(); }

  //! Construit le couple ((ax,bx),(ay,by)).
  constexpr ARCCORE_HOST_DEVICE static Real2x2 fromColumns(Real ax, Real ay, Real bx, Real by)
  {
    return Real2x2(Real2(ax, bx), Real2(ay, by));
  }

  //! Construit le couple ((ax,bx),(ay,by)).
  constexpr ARCCORE_HOST_DEVICE static Real2x2 fromLines(Real ax, Real bx, Real ay, Real by)
  {
    return Real2x2(Real2(ax, bx), Real2(ay, by));
  }

 public:

  //! Retourne une copie du couple.
  constexpr ARCCORE_HOST_DEVICE Real2x2 copy() const { return (*this); }
  //! Réinitialise le couple avec les constructeurs par défaut.
  constexpr ARCCORE_HOST_DEVICE Real2x2& reset()
  {
    *this = null();
    return (*this);
  }
  //! Affecte à l'instance le couple (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real2x2& assign(Real2 ax, Real2 ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }
  //! Copie le couple \a f
  constexpr ARCCORE_HOST_DEVICE Real2x2& assign(Real2x2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  /*!
   * \brief Lit la matrice sur le flot \a i
   * La matrice est lue sous la forme de trois Real2.
   */
  std::istream& assign(std::istream& i);
  //! Ecrit le couple sur le flot \a o lisible par un assign()
  std::ostream& print(std::ostream& o) const;
  //! Ecrit le couple sur le flot \a o sous la forme (x,y,z)
  std::ostream& printXy(std::ostream& o) const;

  //! Ajoute \a b au couple
  constexpr ARCCORE_HOST_DEVICE Real2x2& add(Real2x2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }
  //! Soustrait \a b au couple
  constexpr ARCCORE_HOST_DEVICE Real2x2& sub(Real2x2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  //Real2x2& mul(Real2x2 b) { x*=b.x; y*=b.y; return (*this); }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& div(Real2x2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }
  //! Ajoute \a b à chaque composante du couple
  constexpr ARCCORE_HOST_DEVICE Real2x2& addSame(Real2 b)
  {
    x += b;
    y += b;
    return (*this);
  }
  //! Soustrait \a b à chaque composante du couple
  constexpr ARCCORE_HOST_DEVICE Real2x2& subSame(Real2 b)
  {
    x -= b;
    y -= b;
    return (*this);
  }
  //! Multiplie chaque composante du couple par \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& mulSame(Real2 b)
  {
    x *= b;
    y *= b;
    return (*this);
  }
  //! Divise chaque composante du couple par \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2& divSame(Real2 b)
  {
    x /= b;
    y /= b;
    return (*this);
  }
  //! Ajoute \a b au couple.
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator+=(Real2x2 b) { return add(b); }
  //! Soustrait \a b au couple
  constexpr ARCCORE_HOST_DEVICE Real2x2& operator-=(Real2x2 b) { return sub(b); }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  //Real2x2& operator*=(Real2x2 b) { return mul(b); }
  //! Multiple chaque composante de la matrice par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  //Real2x2& operator/= (Real2x2 b) { return div(b); }
  //! Divise chaque composante de la matrice par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }
  //! Créé un couple qui vaut ce couple ajouté à \a b
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator+(Real2x2 b) const { return Real2x2(x + b.x, y + b.y); }
  //! Créé un couple qui vaut \a b soustrait de ce couple
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator-(Real2x2 b) const { return Real2x2(x - b.x, y - b.y); }
  //! Créé un tenseur opposé au tenseur actuel
  constexpr ARCCORE_HOST_DEVICE Real2x2 operator-() const { return Real2x2(-x, -y); }

  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y.
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real2x2 b) const
  {
    return (x == b.x) && (y == b.y);
  }
  /*!
   * \brief Compare deux couples.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux couples sont différents,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real2x2 b) const
  {
    return !operator==(b);
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème (entre 0 et 1 inclus) ligne de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real2 operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème (entre 0 et 1 inclus) ligne de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real2 operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème ligne et \a j-ème colonne.
   * \param i numéro de la ligne à retourner
   * \param j numéro de la colonne à retourner
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i, Integer j) const
  {
    ARCCORE_CHECK_AT(i, 2);
    ARCCORE_CHECK_AT(j, 2);
    return (&x)[i][j];
  }

  /*!
   * \brief  Accès à la \a i-ème ligne (entre 0 et 1 inclus) de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real2& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief  Accès à la \a i-ème ligne (entre 0 et 1 inclus) de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real2& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * \brief Accès à la \a i-ème ligne et \a j-ème colonne.
   * \param i numéro de la ligne à retourner
   * \param j numéro de la colonne à retourner
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i, Integer j)
  {
    ARCCORE_CHECK_AT(i, 2);
    ARCCORE_CHECK_AT(j, 2);
    return (&x)[i][j];
  }

 public:

  //! Ecrit le couple \a t sur le flot \a o
  friend std::ostream& operator<<(std::ostream& o, Real2x2 t)
  {
    return t.printXy(o);
  }

  //! Lit le couple \a t à partir du flot \a o.
  friend std::istream& operator>>(std::istream& i, Real2x2& t)
  {
    return t.assign(i);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator*(Real sca, Real2x2 vec)
  {
    return Real2x2(vec.x * sca, vec.y * sca);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator*(Real2x2 vec, Real sca)
  {
    return Real2x2(vec.x * sca, vec.y * sca);
  }

  //! Division par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2x2 operator/(Real2x2 vec, Real sca)
  {
    return Real2x2(vec.x / sca, vec.y / sca);
  }

  /*!
  * \brief Opérateur de comparaison.
  *
  * Cet opérateur permet de trier les Real2 pour les utiliser par exemple
  * dans les std::set
  */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real2x2 v1, Real2x2 v2)
  {
    if (v1.x == v2.x) {
      return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

 public:

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
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real2x2&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

 private:

  /*!
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b)
  {
    return TypeEqualT<Real>::isEqual(a, b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
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
  constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real2x2& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real2x2::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
