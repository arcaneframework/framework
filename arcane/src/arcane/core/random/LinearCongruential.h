// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LinearCongruential.h                                        (C) 2000-2015 */
/*                                                                           */
/* Randomiser 'LinearCongruential'.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_LINEARCONGRUENTIAL_H
#define ARCANE_RANDOM_LINEARCONGRUENTIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/String.h"

#include "arcane/core/random/RandomGlobal.h"
#include "arcane/core/random/ConstMod.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief compile-time configurable linear congruential generator.
 *
 * \warning Cette implémentation n'est valide que pour les valeurs (a,c,m)
 * correspondantes à MinstdRand et MinstdRand0.
 *
 */
template<typename IntType, IntType a, IntType c, IntType m, IntType val>
class LinearCongruential
{
 public:
  typedef IntType result_type;
  static const bool has_fixed_range = true;
  static const result_type min_value = ( c == 0 ? 1 : 0 );
  static const result_type max_value = m-1;
  static const IntType multiplier = a;
  static const IntType increment = c;
  static const IntType modulus = m;

  result_type min() const { return c == 0 ? 1 : 0; }
  result_type max() const { return m-1; }
  explicit LinearCongruential(IntType x0 = 1)
    : _x(x0)
  { 
#ifdef ARCANE_CHECK
    checkSeed(_x);
#endif
  }
  // compiler-generated copy constructor and assignment operator are fine
  void seed(IntType x0)
  {
    _x = x0;
#ifdef ARCANE_CHECK
    checkSeed(_x);
#endif
  }
  IntType getState() const { return _x; }
  IntType operator()()
  {
    _x = apply(_x);
    return _x;
  }
  static IntType apply(IntType x)
  {
    return x = utils::const_mod<IntType, m>::mult_add(a, x, c);
  }
  bool validation(IntType x) const { return val == x; }
  bool operator==(const LinearCongruential& rhs) const
    { return _x == rhs._x; }

  inline void checkSeed(IntType x)
  {
    if (_x<min() || _x>=max())
      throw FatalErrorException(A_FUNCINFO,String::format("Invalid seed v={0} min={1} max={2}",x,min(),max()));
  }

 private:

  IntType _x;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef LinearCongruential<Int32,16807,0,2147483647,1043618065> MinstdRand0;
typedef LinearCongruential<Int32,48271,0,2147483647,399268537> MinstdRand;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
