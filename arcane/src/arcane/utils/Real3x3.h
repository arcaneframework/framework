// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Real3x3.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Matrice 3x3 de 'Real'.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_REAL3X3_H
#define ARCANE_UTILS_REAL3X3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure POD pour un Real3x3.
 */
struct Real3x3POD
{
 public:

  Real3POD x;
  Real3POD y;
  Real3POD z;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant une matrice de réel de dimension 3x3.

 La matrice comprend trois composantes \a x, \a y et \a z qui sont du
 type \b Real3. Chaque composante représente une ligne de la matrice.
 Par conséquent, Pour une matrice \a m, m.y.z représente la 2ème ligne
 et la 3ème colonne de la matrice.

 Il est aussi possible d'accéder aux éléments de la matrice à la
 manière d'un tableau. Par exemple m[1][2] représente la 2ème ligne et la 3ème
 colonne de la matrice.

 Les matrices peuvent se construire par ligne en spécifiant les valeurs
 par ligne (fromLines()) ou en spécifiant par colonne (fromColumns()).

 Par exemple:

 * \code
 * Real3x3 matrix;
 * matrix.x.y = 2.0;
 * matrix.y.z = 3.0;
 * matrix.x.x = 5.0;
 * \endcode
*/
class ARCANE_UTILS_EXPORT Real3x3
{
 public:

  //! Construit la matrice avec tous les coefficiants nuls.
  constexpr ARCCORE_HOST_DEVICE Real3x3()
  : x(Real3::zero())
  , y(Real3::zero())
  , z(Real3::zero())
  {}

  //! Construit la matrice avec les lignes (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3x3(Real3 ax, Real3 ay, Real3 az)
  : x(ax)
  , y(ay)
  , z(az)
  {}

  /*!
   * \brief Construit le tenseur ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
   * \deprecated Utiliser Real3x3(Real3 x,Real3 y,Real3 z) à la place.
   */
  ARCANE_DEPRECATED_116 Real3x3(Real ax, Real ay, Real az, Real bx, Real by, Real bz, Real cx, Real cy, Real cz)
  : x(ax, bx, cx)
  , y(ay, by, cy)
  , z(az, bz, cz)
  {}

  //! Construit un triplet identique à \a f
  Real3x3(const Real3x3& f) = default;
  //! Construit un triplet identique à \a f
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(const Real3x3POD& f)
  : x(f.x)
  , y(f.y)
  , z(f.z)
  {}

  //! Construit l'instance avec le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(Real v)
  {
    x = y = z = v;
  }

  //! Construit le triplet ((av[0], av[1], av[2]), (av[3], av[4], av[5]), (av[6], av[7], av[8]))
  constexpr ARCCORE_HOST_DEVICE explicit Real3x3(ConstArrayView<Real> av)
  : x(av[0], av[1], av[2])
  , y(av[3], av[4], av[5])
  , z(av[6], av[7], av[8])
  {}

  //! Opérateur de recopie
  Real3x3& operator=(const Real3x3& f) = default;

  //! Affecte à l'instance le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator=(Real v)
  {
    x = y = z = v;
    return (*this);
  }

 public:

  Real3 x; //!< premier élément du triplet
  Real3 y; //!< premier élément du triplet
  Real3 z; //!< premier élément du triplet

 public:

  //! Construit le tenseur nul.
  constexpr ARCCORE_HOST_DEVICE static Real3x3 null() { return Real3x3(); }

  //! Construit la matrice nulle
  constexpr ARCCORE_HOST_DEVICE static Real3x3 zero() { return Real3x3(); }

  //! Construit la matrice identité
  constexpr ARCCORE_HOST_DEVICE static Real3x3 identity() { return Real3x3(Real3(1.0, 0.0, 0.0), Real3(0.0, 1.0, 0.0), Real3(0.0, 0.0, 1.0)); }

  //! Construit la matrice ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static Real3x3 fromColumns(Real ax, Real ay, Real az, Real bx, Real by, Real bz, Real cx, Real cy, Real cz)
  {
    return Real3x3(Real3(ax, bx, cx), Real3(ay, by, cy), Real3(az, bz, cz));
  }

  //! Construit la matrice ((ax,bx,cx),(ay,by,cy),(az,bz,cz)).
  constexpr ARCCORE_HOST_DEVICE static Real3x3 fromLines(Real ax, Real bx, Real cx, Real ay, Real by, Real cy, Real az, Real bz, Real cz)
  {
    return Real3x3(Real3(ax, bx, cx), Real3(ay, by, cy), Real3(az, bz, cz));
  }

 public:

  //! Retourne une copie de la matrice
  constexpr ARCCORE_HOST_DEVICE Real3x3 copy() const { return (*this); }

  //! Remet à zéro les coefficients de la matrice.
  constexpr ARCCORE_HOST_DEVICE Real3x3& reset()
  {
    *this = zero();
    return (*this);
  }

  //! Affecte à l'instance les lignes (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Real3x3& assign(Real3 ax, Real3 ay, Real3 az)
  {
    x = ax;
    y = ay;
    z = az;
    return (*this);
  }

  //! Copie la matrice \a f
  constexpr ARCCORE_HOST_DEVICE Real3x3& assign(Real3x3 f)
  {
    x = f.x;
    y = f.y;
    z = f.z;
    return (*this);
  }

  //! Retourne une vue sur les neuf élements de la matrice.
  //! [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
  constexpr ARCCORE_HOST_DEVICE ArrayView<Real> view()
  {
    return { 9, &x.x };
  }

  //! Retourne une vue constante sur les neuf élements de la matrice.
  //! [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<Real> constView() const
  {
    return { 9, &x.x };
  }

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
  constexpr ARCCORE_HOST_DEVICE Real3x3& add(Real3x3 b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return (*this);
  }
  //! Soustrait \a b au triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& sub(Real3x3 b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return (*this);
  }
  //! Ajoute \a b à chaque composante du triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& addSame(Real3 b)
  {
    x += b;
    y += b;
    z += b;
    return (*this);
  }
  //! Soustrait \a b à chaque composante du triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& subSame(Real3 b)
  {
    x -= b;
    y -= b;
    z -= b;
    return (*this);
  }
  //! Ajoute \a b au triplet.
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator+=(Real3x3 b) { return add(b); }
  //! Soustrait \a b au triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3& operator-=(Real3x3 b) { return sub(b); }
  //! Multiple chaque composante de la matrice par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(Real b)
  {
    x *= b;
    y *= b;
    z *= b;
  }
  //! Divise chaque composante de la matrice par le réel \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(Real b)
  {
    x /= b;
    y /= b;
    z /= b;
  }
  //! Créé un triplet qui vaut ce triplet ajouté à \a b
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator+(Real3x3 b) const { return Real3x3(x + b.x, y + b.y, z + b.z); }
  //! Créé un triplet qui vaut \a b soustrait de ce triplet
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator-(Real3x3 b) const { return Real3x3(x - b.x, y - b.y, z - b.z); }
  //! Créé un tenseur opposé au tenseur actuel
  constexpr ARCCORE_HOST_DEVICE Real3x3 operator-() const { return Real3x3(-x, -y, -z); }

  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y et this.z==b.z.
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator==(Real3x3 b) const
  {
    return x == b.x && y == b.y && z == b.z;
  }

  /*!
   * \brief Compare deux triplets.
   * Pour la notion d'égalité, voir operator==()
   * \retval true si les deux triplets sont différents,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE bool operator!=(Real3x3 b) const
  {
    return !operator==(b);
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème (entre 0 et 2 inclus) ligne de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real3 operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème (entre 0 et 2 inclus) ligne de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real3 operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Accès en lecture seule à la \a i-ème ligne et \a j-ème colonne.
   * \param i numéro de la ligne à retourner
   * \param j numéro de la colonne à retourner
   */
  ARCCORE_HOST_DEVICE Real operator()(Integer i, Integer j) const
  {
    ARCCORE_CHECK_AT(i, 3);
    ARCCORE_CHECK_AT(j, 3);
    return (&x)[i][j];
  }

  /*!
   * \brief Accès à la \a i-ème ligne (entre 0 et 2 inclus) de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real3& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Accès à la \a i-ème ligne (entre 0 et 2 inclus) de l'instance.
   * \param i numéro de la ligne à retourner
   */
  ARCCORE_HOST_DEVICE Real3& operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, 3);
    return (&x)[i];
  }

  /*!
   * \brief Accès à la \a i-ème ligne et \a j-ème colonne.
   * \param i numéro de la ligne à retourner
   * \param j numéro de la colonne à retourner
   */
  ARCCORE_HOST_DEVICE Real& operator()(Integer i, Integer j)
  {
    ARCCORE_CHECK_AT(i, 3);
    ARCCORE_CHECK_AT(j, 3);
    return (&x)[i][j];
  }

  //! Déterminant de la matrice
  constexpr ARCCORE_HOST_DEVICE Real determinant() const
  {
    return (x.x * (y.y * z.z - y.z * z.y) + x.y * (y.z * z.x - y.x * z.z) + x.z * (y.x * z.y - y.y * z.x));
  }

  //! Ecrit le triplet \a t sur le flot \a o.
  friend std::ostream& operator<<(std::ostream& o, Real3x3 t)
  {
    return t.printXyz(o);
  }

  //! Lit le triplet \a t à partir du flot \a o.
  friend std::istream& operator>>(std::istream& i, Real3x3& t)
  {
    return t.assign(i);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator*(Real sca, Real3x3 vec)
  {
    return Real3x3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator*(Real3x3 vec, Real sca)
  {
    return Real3x3(vec.x * sca, vec.y * sca, vec.z * sca);
  }

  //! Division par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE Real3x3 operator/(Real3x3 vec, Real sca)
  {
    return Real3x3(vec.x / sca, vec.y / sca, vec.z / sca);
  }

  /*!
  * \brief Opérateur de comparaison.
  *
  * Cet opérateur permet de trier les Real3 pour les utiliser par exemple
  * dans les std::set
  */
  friend constexpr ARCCORE_HOST_DEVICE bool operator<(Real3x3 v1, Real3x3 v2)
  {
    if (v1.x == v2.x) {
      if (v1.y == v2.y)
        return v1.z < v2.z;
      else
        return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

 public:

  // TODO: rendre obsolète mi-2025: ARCANE_DEPRECATED_REASON("Y2024: Use math::isNearlyZero(const Real3x3&) instead")
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
   * La matrice est nulle si et seulement si chacune de ses composantes
   * est inférieure à un espilon donné. La valeur de l'epsilon utilisée est celle
   * de float_info<value_type>::nearlyEpsilon():
   * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
   *
   * \retval true si la matrice est égale à la matrice nulle,
   * \retval false sinon.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real3x3& v)
  {
    return isNearlyZero(v.x) && isNearlyZero(v.y) && isNearlyZero(v.z);
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real3x3::
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
