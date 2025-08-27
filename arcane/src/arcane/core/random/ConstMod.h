// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomGlobal.h                                              (C) 2000-2025 */
/*                                                                           */
/* Namespace pour Random.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_CONSTMOD_H
#define ARCANE_CORE_RANDOM_CONSTMOD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/core/random/RandomGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::random
{

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
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*
 * Ce fichier est issu de la bibliotheque boost et est couvert par
 * la licence:
 *
 *    Boost Software License - Version 1.0 - August 17th, 2003
 *
 * Le code de ConstMod est issu de boost version 1.57.0
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace utils {

template<typename IntType> class make_unsigned;
template<> class make_unsigned<Int32>
{
 public:
  typedef UInt32 type;
};
template<> class make_unsigned<UInt32>
{
 public:
  typedef UInt32 type;
};
typedef UInt64 uintmax_t;


template<class IntType, IntType m>
class ConstMod
{
public:
  static IntType apply(IntType x)
  {
    if(((unsigned_m() - 1) & unsigned_m()) == 0)
      return (unsigned_type(x)) & (unsigned_m() - 1);
    else {
      IntType supress_warnings = (m == 0);
      return x % (m + supress_warnings);
    }
  }

  static IntType add(IntType x, IntType c)
  {
    if(((unsigned_m() - 1) & unsigned_m()) == 0)
      return (unsigned_type(x) + unsigned_type(c)) & (unsigned_m() - 1);
    else if(c == 0)
      return x;
    else if(x < m - c)
      return x + c;
    else
      return x - (m - c);
  }

  static IntType mult(IntType a, IntType x)
  {
    if(((unsigned_m() - 1) & unsigned_m()) == 0)
      return unsigned_type(a) * unsigned_type(x) & (unsigned_m() - 1);
    else if(a == 0)
      return 0;
    else if(a == 1)
      return x;
    else if(m <= traits::const_max/a)      // i.e. a*m <= max
      return mult_small(a, x);
    else if(traits::is_signed && (m%a < m/a))
      return mult_schrage(a, x);
    else
      return mult_general(a, x);
  }

  static IntType mult_add(IntType a, IntType x, IntType c)
  {
    if(((unsigned_m() - 1) & unsigned_m()) == 0)
      return (unsigned_type(a) * unsigned_type(x) + unsigned_type(c)) & (unsigned_m() - 1);
    else if(a == 0)
      return c;
    else if(m <= (traits::const_max-c)/a) {  // i.e. a*m+c <= max
      IntType supress_warnings = (m == 0);
      return (a*x+c) % (m + supress_warnings);
    } else
      return add(mult(a, x), c);
  }

  static IntType pow(IntType a, uintmax_t exponent)
  {
      IntType result = 1;
      while(exponent != 0) {
          if(exponent % 2 == 1) {
              result = mult(result, a);
          }
          a = mult(a, a);
          exponent /= 2;
      }
      return result;
  }

  static IntType invert(IntType x)
  { return x == 0 ? 0 : (m == 0? invert_euclidian0(x) : invert_euclidian(x)); }

private:
  typedef integer_traits<IntType> traits;
  typedef typename make_unsigned<IntType>::type unsigned_type;

  ConstMod();      // don't instantiate

  static IntType mult_small(IntType a, IntType x)
  {
    IntType supress_warnings = (m == 0);
    return a*x % (m + supress_warnings);
  }

  static IntType mult_schrage(IntType a, IntType value)
  {
    const IntType q = m / a;
    const IntType r = m % a;

    return sub(a*(value%q), r*(value/q));
  }

  static IntType mult_general(IntType a, IntType b)
  {
    IntType suppress_warnings = (m == 0);
    IntType modulus = m + suppress_warnings;
    if(uintmax_t(modulus) <=
        (::std::numeric_limits< uintmax_t>::max)() / modulus)
    {
      return static_cast<IntType>(uintmax_t(a) * b % modulus);
    } else {
      throw NotImplementedException("Handling mulmod in mult_general");
    }
  }

  static IntType sub(IntType a, IntType b)
  {
    if(a < b)
      return m - (b - a);
    else
      return a - b;
  }

  static unsigned_type unsigned_m()
  {
      if(m == 0) {
          return unsigned_type((std::numeric_limits<IntType>::max)()) + 1;
      } else {
          return unsigned_type(m);
      }
  }

  // invert c in the finite field (mod m) (m must be prime)
  static IntType invert_euclidian(IntType c)
  {
    // we are interested in the gcd factor for c, because this is our inverse
    IntType l1 = 0;
    IntType l2 = 1;
    IntType n = c;
    IntType p = m;
    for(;;) {
      IntType q = p / n;
      l1 += q * l2;
      p -= q * n;
      if(p == 0)
        return l2;
      IntType q2 = n / p;
      l2 += q2 * l1;
      n -= q2 * p;
      if(n == 0)
        return m - l1;
    }
  }

  // invert c in the finite field (mod m) (c must be relatively prime to m)
  static IntType invert_euclidian0(IntType c)
  {
    // we are interested in the gcd factor for c, because this is our inverse
    if(c == 1) return 1;
    IntType l1 = 0;
    IntType l2 = 1;
    IntType n = c;
    IntType p = m;
    IntType max = (std::numeric_limits<IntType>::max)();
    IntType q = max / n;
    l1 += q * l2;
    p = max - q * n + 1;
    for(;;) {
      if(p == 0)
        return l2;
      IntType q2 = n / p;
      l2 += q2 * l1;
      n -= q2 * p;
      if(n == 0)
        return m - l1;
      q = p / n;
      l1 += q * l2;
      p -= q * n;
    }
  }
};

} // namespace utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

