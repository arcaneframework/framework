// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector3.h                                                   (C) 2000-2023 */
/*                                                                           */
/* Vecteur à 3 dimensions.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VECTOR3_H
#define ARCANE_UTILS_VECTOR3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un vecteur de dimension 3 de type \a T
 *
 * Les 3 éléments du vecteur peuvent être accédés par \a x, \a y ou \a z.
 */
template <typename T>
class Vector3
{
 public:

  using ThatClass = Vector3<T>;
  using value_type = T;

 public:

  T x = {};
  T y = {};
  T z = {};

 public:

  //! Construit le vecteur nul.
  constexpr ARCCORE_HOST_DEVICE
  Vector3() = default;

  //! Construit le triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Vector3(const T& ax, const T& ay, const T& az)
  : x(ax)
  , y(ay)
  , z(az)
  {
  }

  //! Construit l'instance avec le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Vector3(const T& v)
  : x(v)
  , y(v)
  , z(v)
  {
  }

  //! Construit l'instance avec le tableau \a v
  constexpr explicit Vector3(const std::array<T, 3>& v)
  : x(v[0])
  , y(v[1])
  , z(v[2])
  {
  }

  //! Construit l'instance avec la liste \a v
  constexpr Vector3(std::initializer_list<T> v)
  {
    _setFromList(v);
  }

  //! Positionne l'instance avec la liste \a v
  constexpr Vector3& operator=(std::initializer_list<T> v)
  {
    _setFromList(v);
    return (*this);
  }

 public:

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator<(const Vector3<T>& v1, const Vector3<T>& v2)
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
  friend std::ostream& operator<<(std::ostream& o, const Vector3<T>& t)
  {
    t._print(o);
    return o;
  }

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator==(const Vector3<T>& v1, const Vector3<T>& v2)
  {
    return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
  }

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator!=(const Vector3<T>& v1, const Vector3<T>& v2)
  {
    return !(v1 == v2);
  }

 public:

  //! Ajoute \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const T& b)
  {
    x += b;
    y += b;
    z += b;
  }
  //! Ajoute \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const ThatClass& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
  }
  //! Soustrait \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const T& b)
  {
    x -= b;
    y -= b;
    z -= b;
  }
  //! Soustrait \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const ThatClass& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
  }
  //! Multiple chaque composante de l'instance par \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(const T& b)
  {
    x *= b;
    y *= b;
    z *= b;
  }
  //! Divise chaque composante de l'instance par \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(const T& b)
  {
    x /= b;
    y /= b;
    z /= b;
  }
  //! Somme composante par composante de \a a et \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x + b.x, a.y + b.y, a.z + b.z);
  }
  //! Retourne \a a en ajoutant \a b à chaque composante
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x + b, a.y + b, a.z + b);
  }
  //! Retourne \a b en ajoutant \a a à chaque composante
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const T& a, const ThatClass& b)
  {
    return ThatClass(a + b.x, a + b.y, a + b.z);
  }
  //! Soustrait chaque composante de \a a par chaque composante de \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x - b.x, a.y - b.y, a.z - b.z);
  }
  //! Soustrait chaque composante de \a a par \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x - b, a.y - b, a.z - b);
  }
  //! Retourne l'opposé de l'instance
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const { return ThatClass(-x, -y, -z); }

  //! Multiplie chaque composante de \a b par \a a
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const T& a, const ThatClass& b)
  {
    return ThatClass(b.x * a, b.y * a, b.z * a);
  }
  //! Multiplie chaque composante de \a a par \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x * b, a.y * b, a.z * b);
  }
  //! Divise chaque composante de \a a par \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator/(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x / b, a.y / b, a.z / b);
  }

 private:

  void _print(std::ostream& o) const
  {
    o << '{' << x << ',' << y << ',' << z << '}';
  }
  constexpr void _setFromList(std::initializer_list<T> v)
  {
    auto s = v.size();
    auto ptr = v.begin();
    if (s > 0)
      x = *ptr;
    ++ptr;
    if (s > 1)
      y = *ptr;
    ++ptr;
    if (s > 2)
      z = *ptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
