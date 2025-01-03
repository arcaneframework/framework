// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3.h                                                     (C) 2000-2024 */
/*                                                                           */
/* Vecteur à 3 dimensions.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3_H
#define ARCANE_UTILS_REAL3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"
#include "arcane/utils/Real2.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct Real3POD
{
 public:

  Real x; //!< première composante du triplet
  Real y; //!< deuxième composante du triplet
  Real z; //!< troisième composante du triplet

  /*!
   * Accès en lecture seule à la @a i eme composante du Real3POD
   *
   * @note ne fonctionne que pour x, y et z ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * Accès en lecture seule à la @a i eme composante du Real3POD
   *
   * @note ne fonctionne que pour x, y et z ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * Accès à la @a i eme composante du Real3POD
   *
   * @note ne fonctionne que pour x, y et z ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,3);
    return (&x)[i];
  }

  /*!
   * Accès à la @a i eme composante du Real3POD
   *
   * @note ne fonctionne que pour x, y et z ordonnées dans le POD
   *
   * @param i numéro de la composante à retourner
   *
   * @return (&x)[i]
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i,3);
    return (&x)[i];
  }

  //! Positionne la \a i-ème composante à \a value
  ARCCORE_HOST_DEVICE void setComponent(Integer i, Real value)
  {
    ARCCORE_CHECK_AT(i, 3);
    (&x)[i] = value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un vecteur de réel de dimension 3.

 Le vecteur comprend trois composantes \a x, \a y et \a z qui sont du
 type \b Real.

 \code
 Real3 value (1.0,2.3,4.5); // Créé un triplet (x=1.0, y=2.3, z=4.5)
 cout << value.x;           // Imprime le composant x 
 value.y += 3.2;            // Ajoute 3.2 à la composante \b y
 \endcode

 ou de maniere equivalente

 \code
 Real3 value (1.0,2.3,4.5); // Créé un triplet (x=1.0, y=2.3, z=4.5)
 cout << value[0];          // Imprime le composant x 
 value[1] += 3.2;           // Ajoute 3.2 à la composante \b y
 \endcode
 */

class ARCANE_UTILS_EXPORT Real3
: public Real3POD
{
 public:

  //! Construit le vecteur nul.
  constexpr ARCCORE_HOST_DEVICE Real3()
  : Real3POD()
  {
    x = 0.0;
    y = 0.0;
    z = 0.0;
  }
  //! Construit le triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3(Real ax, Real ay, Real az)
  : Real3POD()
  {
    x = ax;
    y = ay;
    z = az;
  }
  //! Construit un triplet identique à \a f
  Real3(const Real3& f) = default;
  //! Construit un triplet identique à \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3(const Real3POD& f)
  : Real3POD()
  {
    x = f.x;
    y = f.y;
    z = f.z;
  }

  //! Construit l'instance avec le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real3(Real v)
  : Real3POD()
  {
    x = y = z = v;
  }

  //! Construit un triplet identique à \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3(const Real2& f)
  : Real3POD()
  {
    x = f.x;
    y = f.y;
    z = 0.0;
  }

  //! Opérateur de recopie.
  Real3& operator=(const Real3& f) = default;

  //! Affecte à l'instance le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE Real3& operator=(Real v)
  {
    x = y = z = v;
    return (*this);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE static Real3 null() { return Real3(0., 0., 0.); }
  constexpr ARCCORE_HOST_DEVICE static Real3 zero() { return Real3(0., 0., 0.); }

 public:

  //! Retourne une copie du triplet.
  constexpr ARCCORE_HOST_DEVICE Real3 copy() const { return (*this); }
  //! Réinitialise le triplet avec les constructeurs par défaut.
  constexpr ARCCORE_HOST_DEVICE Real3& reset()
  {
    x = y = z = 0.;
    return (*this);
  }
  //! Affecte à l'instance le triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3& assign(Real ax, Real ay, Real az)
  {
    x = ax;
    y = ay;
    z = az;
    return (*this);
  }
  //! Copie le triplet \a f
  constexpr ARCCORE_HOST_DEVICE Real3& assign(Real3 f)
  {
    x = f.x;
    y = f.y;
    z = f.z;
    return (*this);
  }

  //! Valeur absolue composante par composante.
  ARCCORE_HOST_DEVICE Real3 absolute() const { return Real3(math::abs(x), math::abs(y), math::abs(z)); }

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
  constexpr ARCCORE_HOST_DEVICE Real3& add(Real3 b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return (*this);
  }
  //! Soustrait \a b au triplet
  constexpr ARCCORE_HOST_DEVICE Real3& sub(Real3 b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return (*this);
  }
  //! Multiple chaque composante du triplet par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real3& mul(Real3 b)
  {
    x *= b.x;
    y *= b.y;
    z *= b.z;
    return (*this);
  }
  //! Divise chaque composante du triplet par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real3& div(Real3 b)
  {
    x /= b.x;
    y /= b.y;
    z /= b.z;
    return (*this);
  }
  //! Ajoute \a b à chaque composante du triplet
  constexpr ARCCORE_HOST_DEVICE Real3& addSame(Real b)
  {
    x += b;
    y += b;
    z += b;
    return (*this);
  }
  //! Soustrait \a b à chaque composante du triplet
  constexpr ARCCORE_HOST_DEVICE Real3& subSame(Real b)
  {
    x -= b;
    y -= b;
    z -= b;
    return (*this);
  }
  //! Multiplie chaque composante du triplet par \a b
  constexpr ARCCORE_HOST_DEVICE Real3& mulSame(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
    return (*this);
  }
  //! Divise chaque composante du triplet par \a b
  constexpr ARCCORE_HOST_DEVICE Real3& divSame(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
    return (*this);
  }
  //! Ajoute \a b au triplet.
  constexpr ARCCORE_HOST_DEVICE Real3& operator+=(Real3 b) { return add(b); }
  //! Soustrait \a b au triplet
  constexpr ARCCORE_HOST_DEVICE Real3& operator-=(Real3 b) { return sub(b); }
  //! Multiple chaque composante du triplet par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real3& operator*=(Real3 b) { return mul(b); }
  //! Multiple chaque composante du triplet par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
  }
  //! Divise chaque composante du triplet par la composant correspondant de \a b
  constexpr ARCCORE_HOST_DEVICE Real3& operator/=(Real3 b) { return div(b); }
  //! Divise chaque composante du triplet par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
  }
  //! Créé un triplet qui vaut ce triplet ajouté à \a b
  constexpr ARCCORE_HOST_DEVICE Real3 operator+(Real3 b) const { return Real3(x + b.x, y + b.y, z + b.z); }
  //! Créé un triplet qui vaut \a b soustrait de ce triplet
  constexpr ARCCORE_HOST_DEVICE Real3 operator-(Real3 b) const { return Real3(x - b.x, y - b.y, z - b.z); }
  //! Créé un triplet opposé au triplet actuel
  constexpr ARCCORE_HOST_DEVICE Real3 operator-() const { return Real3(-x, -y, -z); }
  /*!
   * \brief Créé un triplet qui vaut ce triplet dont chaque composant a été
   * multipliée par la composante correspondante de \a b.
   */
  constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real3 b) const { return Real3(x * b.x, y * b.y, z * b.z); }
  /*!
   * \brief Créé un triplet qui vaut ce triplet dont chaque composant a été divisée
   * par la composante correspondante de \a b.
   */
  constexpr ARCCORE_HOST_DEVICE Real3 operator/(Real3 b) const { return Real3(x / b.x, y / b.y, z / b.z); }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real sca, Real3 vec)
  {
    return Real3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator*(Real3 vec, Real sca)
  {
    return Real3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Division par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3 operator/(Real3 vec, Real sca)
  {
    return Real3(vec.x / sca, vec.y / sca, vec.z / sca);
  }

 public:

  /*!
   * \brief Opérateur de comparaison.
   *
   * Cet opérateur permet de trier les Real3 pour les utiliser par exemple
   * dans les std::set
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real3 v1, Real3 v2)
  {
    if (v1.x == v2.x) {
      if (v1.y == v2.y)
        return v1.z < v2.z;
      else
        return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

  //! Ecrit le triplet \a t sur le flot \a o
  friend std::ostream& operator<<(std::ostream& o, Real3 t)
  {
    return t.printXyz(o);
  }

  //! Lit le triplet \a t à partir du flot \a o.
  friend std::istream& operator>>(std::istream& i, Real3& t)
  {
    return t.assign(i);
  }

  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y et this.z==b.z.
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real3 b) const
  {
    return _eq(x, b.x) && _eq(y, b.y) && _eq(z, b.z);
  }

  /*!
   * \brief Compare deux triplets.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux triplets sont différents,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real3 b) const { return !operator==(b); }

 public:

  //! Retourne la norme L2 au carré du triplet \f$x^2+y^2+z^2\f$
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::squareNormL2(const Real3&) instead")
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const { return x * x + y * y + z * z; }

  //! Retourne la norme L2 du triplet \f$\sqrt{x^2+y^2+z^2}\f$
  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::normL2(const Real3&) instead")
  inline ARCCORE_HOST_DEVICE Real normL2() const;

  //! Retourne la norme au carré du triplet \f$x^2+y^2+z^2\f$
  ARCCORE_DEPRECATED_2021("Use math::squareNormL2(const Real3&) instead")
  constexpr ARCCORE_HOST_DEVICE Real abs2() const { return x * x + y * y + z * z; }

  //! Retourne la norme du triplet \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_DEPRECATED_2021("Use math::normL2(const Real3&) instead")
  inline ARCCORE_HOST_DEVICE Real abs() const;

  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real3&) instead")
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const;

  ARCANE_DEPRECATED_REASON("Y2024: Use math::mutableNormalize(Real3&) instead")
  inline Real3& normalize();

 private:

  /*!
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  inline constexpr ARCCORE_HOST_DEVICE static bool _eq(Real a, Real b);
  //! Retourne la racine carrée de \a a
  inline ARCCORE_HOST_DEVICE static Real _sqrt(Real a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Real2::
Real2(const Real3& v)
: Real2POD()
{
  x = v.x;
  y = v.y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
  //! Retourne la norme au carré du triplet \f$x^2+y^2+z^2\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real3& v)
  {
    return v.x * v.x + v.y * v.y + v.z * v.z;
  }

  /*!
   * \brief Indique si l'instance est proche de l'instance nulle.
   *
   * \retval true si math::isNearlyZero() est vrai pour chaque composante.
   * \retval false sinon.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real3& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y) && math::isNearlyZero(v.z);
  }
  //! Retourne la norme L2 du triplet \f$\sqrt{v.x^2+v.y^2+v.z^2}\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real3& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }
  /*!
    * \brief Normalise le triplet \a v
    *
    * Si le triplet est non nul, divise chaque composante par la norme du triplet
    * (abs()), de telle sorte qu'après l'appel à cette méthode, math::normL2() vaux \a 1.
    * Si le triplet est nul, ne fait rien.
    */
  inline Real3& mutableNormalize(Real3& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Real3& Real3::
normalize()
{
  return math::mutableNormalize(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real3::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real3::
_eq(Real a, Real b)
{
  return math::isEqual(a, b);
}

ARCCORE_HOST_DEVICE inline Real Real3::
_sqrt(Real a)
{
  return math::sqrt(a);
}

inline ARCCORE_HOST_DEVICE Real Real3::
normL2() const
{
  return math::normL2(*this);
}

inline ARCCORE_HOST_DEVICE Real Real3::
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
