// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real2.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Vecteur à 2 dimensions de 'Real'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL2_H
#define ARCANE_UTILS_REAL2_H
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

  /*!
   * Accès en lecture seule à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Accès en lecture seule à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Accès à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  /*!
   * Accès à la @a i eme composante du Real2POD
   *
   * @note ne fonctionne que pour x, y ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 2);
    return (&x)[i];
  }

  //! Positionne la \a i-ème composante à \a value
  ARCCORE_HOST_DEVICE void setComponent(Integer i, Real value)
  {
    ARCCORE_CHECK_AT(i, 2);
    (&x)[i] = value;
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
  constexpr ARCCORE_HOST_DEVICE Real2()
  : Real2POD()
  {
    x = 0.;
    y = 0.;
  }
  //! Construit le couplet (ax,ay)
  constexpr ARCCORE_HOST_DEVICE Real2(Real ax, Real ay)
  : Real2POD()
  {
    x = ax;
    y = ay;
  }
  //! Construit un couple identique à \a f
  Real2(const Real2& f) = default;
  //! Construit un coupe identique à \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real2(const Real2POD& f)
  : Real2POD()
  {
    x = f.x;
    y = f.y;
  }

  //! Construit l'instance avec le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real2(Real v)
  : Real2POD()
  {
    x = y = v;
  }

  //! Construit l'instance en premier les deux premières composantes de Real3.
  inline constexpr ARCCORE_HOST_DEVICE explicit Real2(const Real3& v);

  //! Construit le couplet (av[0], av[1])
  constexpr ARCCORE_HOST_DEVICE Real2(ConstArrayView<Real> av)
  : Real2POD()
  {
    x = av[0];
    y = av[1];
  }

  Real2& operator=(const Real2& f) = default;

  //! Affecte à l'instance le couple (v,v).
  constexpr ARCCORE_HOST_DEVICE Real2& operator=(Real v)
  {
    x = y = v;
    return (*this);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE static Real2 null() { return Real2(0., 0.); }

 public:

  //! Retourne une copie du couple.
  constexpr ARCCORE_HOST_DEVICE Real2 copy() const { return (*this); }
  //! Réinitialise le couple avec les constructeurs par défaut.
  constexpr ARCCORE_HOST_DEVICE Real2& reset()
  {
    x = y = 0.0;
    return (*this);
  }
  //! Affecte à l'instance le couple (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real2& assign(Real ax, Real ay)
  {
    x = ax;
    y = ay;
    return (*this);
  }
  //! Copie le couple \a f
  constexpr ARCCORE_HOST_DEVICE Real2& assign(Real2 f)
  {
    x = f.x;
    y = f.y;
    return (*this);
  }

  //! Retourne une vue sur les deux élements du vecteur.
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 2, &x };
  }

  //! Retourne une vue constante sur les deux élements du vecteur.
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 2, &x };
  }

  //! Valeur absolue composante par composante.
  ARCCORE_HOST_DEVICE Real2 absolute() const { return Real2(math::abs(x), math::abs(y)); }

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
  constexpr ARCCORE_HOST_DEVICE Real2& add(Real2 b)
  {
    x += b.x;
    y += b.y;
    return (*this);
  }
  //! Soustrait \a b au couple
  constexpr ARCCORE_HOST_DEVICE Real2& sub(Real2 b)
  {
    x -= b.x;
    y -= b.y;
    return (*this);
  }
  //! Multiple chaque composante du couple par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real2& mul(Real2 b)
  {
    x *= b.x;
    y *= b.y;
    return (*this);
  }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real2& div(Real2 b)
  {
    x /= b.x;
    y /= b.y;
    return (*this);
  }
  //! Ajoute \a b à chaque composante du couple
  constexpr ARCCORE_HOST_DEVICE Real2& addSame(Real b)
  {
    x += b;
    y += b;
    return (*this);
  }
  //! Soustrait \a b à chaque composante du couple
  constexpr ARCCORE_HOST_DEVICE Real2& subSame(Real b)
  {
    x -= b;
    y -= b;
    return (*this);
  }
  //! Multiplie chaque composante du couple par \a b
  constexpr ARCCORE_HOST_DEVICE Real2& mulSame(Real b)
  {
    x *= b;
    y *= b;
    return (*this);
  }
  //! Divise chaque composante du couple par \a b
  constexpr ARCCORE_HOST_DEVICE Real2& divSame(Real b)
  {
    x /= b;
    y /= b;
    return (*this);
  }
  //! Ajoute \a b au couple.
  constexpr ARCCORE_HOST_DEVICE Real2& operator+=(Real2 b) { return add(b); }
  //! Soustrait \a b au couple
  constexpr ARCCORE_HOST_DEVICE Real2& operator-=(Real2 b) { return sub(b); }
  //! Multiplie chaque composante du couple par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real2& operator*=(Real2 b) { return mul(b); }
  //! Multiplie chaque composante du couple par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
  }
  //! Divise chaque composante du couple par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real2& operator/=(Real2 b) { return div(b); }
  //! Divise chaque composante du couple par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
  }
  //! Créé un couple qui vaut ce couple ajouté à \a b
  constexpr ARCCORE_HOST_DEVICE Real2 operator+(Real2 b) const { return Real2(x + b.x, y + b.y); }
  //! Créé un couple qui vaut \a b soustrait de ce couple
  constexpr ARCCORE_HOST_DEVICE Real2 operator-(Real2 b) const { return Real2(x - b.x, y - b.y); }
  //! Créé un couple opposé au couple actuel
  constexpr ARCCORE_HOST_DEVICE Real2 operator-() const { return Real2(-x, -y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real2 b) const { return Real2(x * b.x, y * b.y); }
  /*!
   * \brief Créé un couple qui vaut ce couple dont chaque composant a été divisée
   * par la composante correspondante de \a b.
   */
  constexpr ARCCORE_HOST_DEVICE Real2 operator/(Real2 b) const { return Real2(x / b.x, y / b.y); }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real sca, Real2 vec)
  {
    return Real2(vec.x * sca, vec.y * sca);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator*(Real2 vec, Real sca)
  {
    return Real2(vec.x * sca, vec.y * sca);
  }

  //! Division par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real2 operator/(Real2 vec, Real sca)
  {
    return Real2(vec.x / sca, vec.y / sca);
  }

  /*!
   * \brief Opérateur de comparaison.
   *
   * Cet opérateur permet de trier les Real2 pour les utiliser par exemple
   * dans les std::set
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real2 v1, Real2 v2)
  {
    if (v1.x == v2.x) {
      return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

  //! Ecrit le couple \a t sur le flot \a o.
  friend std::ostream& operator<<(std::ostream& o, Real2 t)
  {
    return t.printXy(o);
  }

  //! Lit le couple \a t à partir du flot \a o.
  friend std::istream& operator>>(std::istream& i, Real2& t)
  {
    return t.assign(i);
  }

  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y.
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real2 b) const
  {
    return _eq(x, b.x) && _eq(y, b.y);
  }

  /*!
   * \brief Compare deux couples.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux couples sont différents,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real2 b) const { return !operator==(b); }

 public:

  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::squareNormL2(*this) instead")
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const { return x * x + y * y; }

  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  ARCCORE_DEPRECATED_2021("Use math::squareNormL2(*this) instead")
  ARCCORE_HOST_DEVICE Real abs2() const { return x * x + y * y; }

  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_DEPRECATED_2021("Use math::normL2(*this) instead")
  inline ARCCORE_HOST_DEVICE Real abs() const;

  /*!
   * \brief Indique si l'instance est proche de l'instance nulle.
   *
   * \retval true si math::isNearlyZero() est vrai pour chaque composante.
   * \retval false sinon.
   */
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real2&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::normL2(const Real2&) instead")
  ARCCORE_HOST_DEVICE Real normL2() const;

  /*!
   * \brief Normalise le couple.
   *
   * Si le couple est non nul, divise chaque composante par la norme du couple
   * (abs()), de telle sorte qu'après l'appel à cette méthode, abs() valent \a 1.
   * Si le couple est nul, ne fait rien.
   */
  ARCANE_DEPRECATED_REASON("Y2024: Use math::mutableNormalize(Real2&) instead")
  inline Real2& normalize();

 private:

  /*!
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b);
  //! Retourne la racine carrée de \a a
  ARCCORE_HOST_DEVICE static Real _sqrt(Real a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
  /*!
   * \brief Indique si l'instance est proche de l'instance nulle.
   *
   * \retval true si math::isNearlyZero() est vrai pour chaque composante.
   * \retval false sinon.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real2& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y);
  }

  //! Retourne la norme au carré du couple \f$x^2+y^2+z^2\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real2& v)
  {
    return v.x * v.x + v.y * v.y;
  }

  //! Retourne la norme du couple \f$\sqrt{x^2+y^2+z^2}\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real2& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }

  /*!
   * \brief Normalise le couple.
   *
   * Si le couple est non nul, divise chaque composante par la norme du couple
   * (abs()), de telle sorte qu'après l'appel à cette méthode, abs() valent \a 1.
   * Si le couple est nul, ne fait rien.
   */
  inline Real2& mutableNormalize(Real2& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real2::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real2::
_eq(Real a, Real b)
{
  return math::isEqual(a, b);
}

inline ARCCORE_HOST_DEVICE Real Real2::
_sqrt(Real a)
{
  return math::sqrt(a);
}

inline ARCCORE_HOST_DEVICE Real Real2::
normL2() const
{
  return math::normL2(*this);
}

inline Real2& Real2::
normalize()
{
  return math::mutableNormalize(*this);
}

inline ARCCORE_HOST_DEVICE Real Real2::
abs() const
{
  return math::normL2(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
