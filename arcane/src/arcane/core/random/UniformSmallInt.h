// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UniformSmallInt.h                                           (C) 2000-2025 */
/*                                                                           */
/* Randomiser 'UniformSmallInt'.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_UNIFORMONSMALLINT_H
#define ARCANE_CORE_RANDOM_UNIFORMONSMALLINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/NormalDistribution.h"

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
 * $Id: UniformSmallInt.h 3932 2004-08-23 08:45:03Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// uniform integer distribution on a small range [min, max]
template<class UniformRandomNumberGenerator, class IntType = int>
class UniformSmallInt
{
public:
  typedef UniformRandomNumberGenerator base_type;
  typedef IntType result_type;
  static const bool has_fixed_range = false;

  UniformSmallInt(base_type & rng, IntType min, IntType max);
  result_type operator()()
  {
    // we must not use the low bits here, because LCGs get very bad then
    return ((_rng() - _rng.min()) / _factor) % _range + _min;
  }
  result_type min() const { return _min; }
  result_type max() const { return _max; }
private:
  typedef typename base_type::result_type base_result;
  base_type & _rng;
  IntType _min, _max;
  base_result _range;
  int _factor;
};

template<class UniformRandomNumberGenerator, class IntType>
UniformSmallInt<UniformRandomNumberGenerator, IntType>::
UniformSmallInt(base_type & rng, IntType min, IntType max) 
  : _rng(rng), _min(min), _max(max),
    _range(static_cast<base_result>(_max-_min)+1), _factor(1)
{
  // LCGs get bad when only taking the low bits.
  // (probably put this logic into a partial template specialization)
  // Check how many low bits we can ignore before we get too much
  // quantization error.
  base_result r_base = _rng.max() - _rng.min();
  if(r_base == std::numeric_limits<base_result>::max()) {
    _factor = 2;
    r_base /= 2;
  }
  r_base += 1;
  if(r_base % _range == 0) {
    // No quantization effects, good
    _factor = r_base / _range;
  } else {
    // carefully avoid overflow; pessimizing heree
    for( ; r_base/_range/32 >= _range; _factor *= 2)
      r_base /= 2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
