// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomGlobal.h                                              (C) 2000-2017 */
/*                                                                           */
/* Namespace pour Random.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_RANDOMGLOBAL_H
#define ARCANE_RANDOM_RANDOMGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include <climits>
#include <limits>

#define RANDOM_BEGIN_NAMESPACE  namespace random {
#define RANDOM_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
RANDOM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*
 * Copyright Jens Maurer 2000-2001
 * Permission to use, copy, modify, sell, and distribute this software
 * is hereby granted without fee provided that the above copyright notice
 * appears in all copies and that both that copyright notice and this
 * permission notice appear in supporting documentation,
 *
 * Jens Maurer makes no representations about the suitability of this
 * software for any purpose. It is provided "as is" without express or
 * implied warranty.
 *
 * See http://www.boost.org for most recent version including documentation.
 *
 * $Id: RandomGlobal.h 3932 2004-08-23 08:45:03Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace utils {
template<class T, T min_val, T max_val>
class integer_traits_base
{
 public:
  static const bool is_integral = true;
  static const T const_min = min_val;
  static const T const_max = max_val;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class integer_traits;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class integer_traits<int>
  : public std::numeric_limits<int>,
    public integer_traits_base<int, INT_MIN, INT_MAX>
{ };

template<>
class integer_traits<unsigned int>
  : public std::numeric_limits<unsigned int>,
    public integer_traits_base<unsigned int, 0, UINT_MAX>
{ };

template<>
class integer_traits<long>
  : public std::numeric_limits<long>,
    public integer_traits_base<long, LONG_MIN, LONG_MAX>
{ };

template<>
class integer_traits<unsigned long>
  : public std::numeric_limits<unsigned long>,
    public integer_traits_base<unsigned long, 0, ULONG_MAX>
{ };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<bool is_signed>
struct do_add
{ };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
struct do_add<true>
{
  template<class IntType>
  static IntType add(IntType m, IntType x, IntType c)
    {
      x += (c-m);
      if(x < 0)
        x += m;
      return x;
    }
};

template<>
struct do_add<false>
{
  template<class IntType>
  static IntType add(IntType, IntType, IntType)
    {
      // difficult
      throw FatalErrorException("const_mod::add with c too large");
      return 0;
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class IntType, IntType m>
class const_mod
{
public:
  static IntType add(IntType x, IntType c)
  {
    if(c == 0)
      return x;
    else if(c <= traits::const_max - m)    // i.e. m+c < max
      return add_small(x, c);
    else
      return do_add<traits::is_signed>::add(m, x, c);
  }

  static IntType mult(IntType a, IntType x)
  {
    if(a == 1)
      return x;
    else if(m <= traits::const_max/a)      // i.e. a*m <= max
      return mult_small(a, x);
    else if(traits::is_signed && (m%a < m/a))
      return mult_schrage(a, x);
    else {
      // difficult
#ifdef ARCANE_CHECK
      throw FatalErrorException("const_mod::mult with a too large");
#endif
      return 0;
    }
  }

  static IntType mult_add(IntType a, IntType x, IntType c)
  {
    if(m <= (traits::const_max-c)/a)   // i.e. a*m+c <= max
      return (a*x+c) % m;
    else
      return add(mult(a, x), c);
  }

  static IntType invert(IntType x)
  { return x == 0 ? 0 : invert_euclidian(x); }

private:
  typedef integer_traits<IntType> traits;

  const_mod();      // don't instantiate

  static IntType add_small(IntType x, IntType c)
  {
    x += c;
    if(x >= m)
      x -= m;
    return x;
  }

  static IntType mult_small(IntType a, IntType x)
  {
    return a*x % m;
  }

  static IntType mult_schrage(IntType a, IntType value)
  {
    const IntType q = m / a;
    const IntType r = m % a;

    value = a*(value%q) - r*(value/q);
    while(value <= 0)
      value += m;
    return value;
  }

  // invert c in the finite field (mod m) (m must be prime)
  static IntType invert_euclidian(IntType c)
  {
    IntType l1 = 0;
    IntType l2 = 1;
    IntType n = c;
    IntType p = m;
    for(;;) {
      IntType q = p / n;
      l1 -= q * l2;           // this requires a signed IntType!
      p -= q * n;
      if(p == 0)
        return (l2 < 1 ? l2 + m : l2);
      IntType q2 = n / p;
      l2 -= q2 * l1;
      n -= q2 * p;
      if(n == 0)
        return (l1 < 1 ? l1 + m : l1);
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The modulus is exactly the word size: rely on machine overflow handling.
// Due to a GCC bug, we cannot partially specialize in the presence of
// template value parameters.
template<>
class const_mod<unsigned int, 0>
{
  typedef unsigned int IntType;
public:
  static IntType add(IntType x, IntType c) { return x+c; }
  static IntType mult(IntType a, IntType x) { return a*x; }
  static IntType mult_add(IntType a, IntType x, IntType c) { return a*x+c; }

  // m is not prime, thus invert is not useful
private:                      // don't instantiate
  const_mod();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class const_mod<unsigned long, 0>
{
  typedef unsigned long IntType;
public:
  static IntType add(IntType x, IntType c) { return x+c; }
  static IntType mult(IntType a, IntType x) { return a*x; }
  static IntType mult_add(IntType a, IntType x, IntType c) { return a*x+c; }

  // m is not prime, thus invert is not useful
private:                      // don't instantiate
  const_mod();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Correctly compare two numbers whose types possibly differ in signedness.
 * See boost::numeric_cast<> for the general idea.
 * Most "if" statements involve only compile-time constants, so the
 * optimizing compiler can do its job easily.
 *
 * With most compilers, the straightforward implementation produces a
 * bunch of (legitimate) warnings.  Some template magic helps, though.
 */

template<bool signed1, bool signed2>
struct do_compare
{ };

template<>
struct do_compare<false, false>
{
  // cast to the larger type is automatic with built-in types
  template<class T1, class T2>
  static bool equal(T1 x, T2 y) { return x == y; }
  template<class T1, class T2>
  static bool lessthan(T1 x, T2 y) { return x < y; }
};

template<>
struct do_compare<true, true> : public do_compare<false, false>
{ };

template<>
struct do_compare<true, false>
{
  template<class T1, class T2>
  static bool equal(T1 x, T2 y) { return x >= 0 && static_cast<T2>(x) == y; }
  template<class T1, class T2>
  static bool lessthan(T1 x, T2 y) { return x < 0 || static_cast<T2>(x) < y; }
};

template<>
struct do_compare<false, true>
{
  template<class T1, class T2>
  static bool equal(T1 x, T2 y) { return y >= 0 && x == static_cast<T1>(y); }
  template<class T1, class T2>
  static bool lessthan(T1 x, T2 y) { return y >= 0 && x < static_cast<T1>(y); }
};

template<class T1, class T2>
int equal_signed_unsigned(T1 x, T2 y)
{
  typedef std::numeric_limits<T1> x_traits;
  typedef std::numeric_limits<T2> y_traits;
  return do_compare<x_traits::is_signed, y_traits::is_signed>::equal(x, y);
}

template<class T1, class T2>
int lessthan_signed_unsigned(T1 x, T2 y)
{
  typedef std::numeric_limits<T1> x_traits;
  typedef std::numeric_limits<T2> y_traits;
  return do_compare<x_traits::is_signed, y_traits::is_signed>::lessthan(x, y);
}

} // namespace utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

