// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector2.h                                                   (C) 2000-2023 */
/*                                                                           */
/* 2-dimensional vector.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VECTOR2_H
#define ARCANE_UTILS_VECTOR2_H
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
 * \brief Class managing a 2-dimensional vector of type \a T
 *
 * The 2 elements of the vector can be accessed by \a x or \a y.
 */
template <typename T>
class Vector2
{
 public:

  using ThatClass = Vector2<T>;
  using value_type = T;

 public:

  T x = {};
  T y = {};

 public:

  //! Constructs the zero vector.
  constexpr ARCCORE_HOST_DEVICE
  Vector2() = default;

  //! Constructs the triplet (ax,ay,az)
  constexpr ARCCORE_HOST_DEVICE Vector2(const T& ax, const T& ay)
  : x(ax)
  , y(ay)
  {
  }

  //! Constructs the instance with the triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE explicit Vector2(const T& v)
  : x(v)
  , y(v)
  {
  }

  //! Constructs the instance with the array \a v
  constexpr Vector2(const std::array<T, 2>& v)
  : x(v[0])
  , y(v[1])
  {
  }

  //! Constructs the instance with the list \a v
  constexpr Vector2(std::initializer_list<T> v)
  {
    _setFromList(v);
  }

  //! Positions the instance with the list \a v
  constexpr Vector2& operator=(std::initializer_list<T> v)
  {
    _setFromList(v);
    return (*this);
  }

 public:

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator<(const Vector2<T>& v1, const Vector2<T>& v2)
  {
    if (v1.x == v2.x) {
      return v1.y < v2.y;
    }
    return (v1.x < v2.x);
  }

  //! Writes \a t to the stream \a o
  friend std::ostream& operator<<(std::ostream& o, const Vector2<T>& t)
  {
    t._print(o);
    return o;
  }

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator==(const Vector2<T>& v1, const Vector2<T>& v2)
  {
    return v1.x == v2.x && v1.y == v2.y;
  }

  friend constexpr ARCCORE_HOST_DEVICE bool
  operator!=(const Vector2<T>& v1, const Vector2<T>& v2)
  {
    return !(v1 == v2);
  }

 public:

  //! Adds \a b to the instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const T& b)
  {
    x += b;
    y += b;
  }

  //! Adds \a b to the instance
  constexpr ARCCORE_HOST_DEVICE void operator+=(const ThatClass& b)
  {
    x += b.x;
    y += b.y;
  }

  //! Subtracts \a b from the instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const T& b)
  {
    x -= b;
    y -= b;
  }

  //! Subtracts \a b from the instance
  constexpr ARCCORE_HOST_DEVICE void operator-=(const ThatClass& b)
  {
    x -= b.x;
    y -= b.y;
  }

  //! Multiplies each component of the instance by \a b
  constexpr ARCCORE_HOST_DEVICE void operator*=(const T& b)
  {
    x *= b;
    y *= b;
  }

  //! Divides each component of the instance by \a b
  constexpr ARCCORE_HOST_DEVICE void operator/=(const T& b)
  {
    x /= b;
    y /= b;
  }

  //! Sums component by component of \a a and \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x + b.x, a.y + b.y);
  }

  //! Returns \a a by adding \a b to each component
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x + b, a.y + b);
  }

  //! Returns \a b by adding \a a to each component
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const T& a, const ThatClass& b)
  {
    return ThatClass(a + b.x, a + b.y);
  }

  //! Subtracts each component of \a a by each component of \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    return ThatClass(a.x - b.x, a.y - b.y);
  }

  //! Subtracts each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x - b, a.y - b);
  }

  //! Returns the opposite of the instance
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const { return ThatClass(-x, -y); }

  //! Multiplies each component of \a b by \a a
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const T& a, const ThatClass& b)
  {
    return ThatClass(b.x * a, b.y * a);
  }

  //! Multiplies each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x * b, a.y * b);
  }

  //! Divides each component of \a a by \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator/(const ThatClass& a, const T& b)
  {
    return ThatClass(a.x / b, a.y / b);
  }

 private:

  void _print(std::ostream& o) const
  {
    o << '{' << x << ',' << y << '}';
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
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
