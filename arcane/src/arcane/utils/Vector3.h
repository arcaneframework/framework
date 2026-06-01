// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector3.h                                                   (C) 2000-2023 */
/*                                                                           */
/* Vector in 3 dimensions.                                                   */
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
 * \brief Class managing a 3-dimensional vector of type \a T
 *
 * The 3 elements of the vector can be accessed via \a x, \a y, or \a z.
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

  //! Constructs the zero vector.
  constexpr ARCCORE_HOST_DEVICE
  Vector3() = default;

  //! Constructs the triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Vector3(const T& ax, const T& ay, const T& az)
  : x(ax)
  , y(ay)
  , z(az)
  {
  }

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Vector3(const T& v)
  : x(v)
  , y(v)
  , z(v)
  {
  }

  //! Constructs the instance with the array \a v
  constexpr explicit Vector3(const std::array<T, 3>& v)
  : x(v[0])
  , y(v[1])
  , z(v[2])
  {
  }

  //! Constructs the instance with the list \a v
  constexpr Vector3(std::initializer_list<T> v)
  {
    _setFromList(v);
  }

  //! Positions the instance with the list \a v
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

  //! Writes the triplet \a t to the stream \a o
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

  //! Adds \a b to the instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const T& b)
  {
    x += b;
    y += b;
    z += b;
  }

  //! Adds \a b to the instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const ThatClass& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
  }

  //! Subtracts \a b from the instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const T& b)
  {
    x -= b;
    y -= b;
    z -= b;
  }

  //! Subtracts \a b from the instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const ThatClass& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
  }

  //! Multiplies each component of the instance by \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(const T& b)
  {
    x *= b;
    y *= b;
    z *= b;
  }

  //! Divides each component of the instance by \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(const T& b)
  {
    x /= b;
    y /= b;
    z /= b;
  }

  //! Sums component by component of \a a and \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  //! Returns \a a by adding \a b to each component
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x + b, a.y + b, a.z + b);
  }

  //! Returns \a b by adding \a a to each component
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const T& a, const ThatClass& b)
  {
    return ThatClass(a + b.x, a + b.y, a + b.z);
  }

  //! Subtracts each component of \a a by each component of \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  //! Subtracts each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x - b, a.y - b, a.z - b);
  }

  //! Returns the opposite of the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const { return ThatClass(-x, -y, -z); }


  //! Multiplies each component of \a b by \a a
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const T& a, const ThatClass& b)
  {
    return ThatClass(b.x * a, b.y * a, b.z * a);
  }

  //! Multiplies each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x * b, a.y * b, a.z * b);
  }

  //! Divides each component of \a a by \a b
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
