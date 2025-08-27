// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UniformInt.h                                                (C) 2000-2004 */
/*                                                                           */
/* Randomiser 'UniformInt'.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_UNIFORMONINT_H
#define ARCANE_RANDOM_UNIFORMONINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/UniformSmallInt.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
 * $Id: UniformInt.h 3932 2004-08-23 08:45:03Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// uniform integer distribution on [min, max]
template<class UniformRandomNumberGenerator,class IntType = int>
class UniformInt
{
public:
  typedef UniformRandomNumberGenerator base_type;
  typedef IntType result_type;
  static const bool has_fixed_range = false;

  UniformInt(base_type & rng, IntType min, IntType max) 
    : _rng(rng), _min(min), _max(max), _range(_max - _min),
      _bmin(_rng.min()), _brange(_rng.max() - _bmin)
  {
    if(utils::equal_signed_unsigned(_brange, _range))
      _range_comparison = 0;
    else if(utils::lessthan_signed_unsigned(_brange, _range))
      _range_comparison = -1;
    else
      _range_comparison = 1;
  }
  result_type operator()();
  result_type min() const { return _min; }
  result_type max() const { return _max; }
 private:
  typedef typename base_type::result_type base_result;
  base_type & _rng;
  result_type _min, _max, _range;
  base_result _bmin, _brange;
  int _range_comparison;
};

template<class UniformRandomNumberGenerator, class IntType>
inline IntType UniformInt<UniformRandomNumberGenerator, IntType>::operator()()
{
  if(_range_comparison == 0) {
    // this will probably never happen in real life
    // basically nothing to do; just take care we don't overflow / underflow
    return static_cast<result_type>(_rng() - _bmin) + _min;
  } else if(_range_comparison < 0) {
    // use rejection method to handle things like 0..3 --> 0..4
    for(;;) {
      // concatenate several invocations of the base RNG
      // take extra care to avoid overflows
      result_type limit;
      if(_range == std::numeric_limits<result_type>::max()) {
        limit = _range/(static_cast<result_type>(_brange)+1);
        if(_range % static_cast<result_type>(_brange)+1 == static_cast<result_type>(_brange))
          ++limit;
      } else {
        limit = (_range+1)/(static_cast<result_type>(_brange)+1);
      }
      // we consider "result" as expressed to base (_brange+1)
      // for every power of (_brange+1), we determine a random factor
      result_type result = 0;
      result_type mult = 1;
      while(mult <= limit) {
        result += (_rng() - _bmin) * mult;
        mult *= static_cast<result_type>(_brange)+1;
      }
      if(mult == limit)
        // _range+1 is an integer power of _brange+1: no rejections required
        return result;
      // _range/mult < _brange+1  -> no endless loop
      result += UniformInt<base_type,result_type>(_rng, 0, _range/mult)() * mult;
      if(result <= _range)
        return result + _min;
    }
  } else {                   // brange > range
    if(_brange / _range > 4 /* quantization_cutoff */ ) {
      // the new range is vastly smaller than the source range,
      // so quantization effects are not relevant
      return UniformSmallInt<base_type,result_type>(_rng, _min, _max)();
    } else {
      // use rejection method to handle things like 0..5 -> 0..4
      for(;;) {
        base_result result = _rng() - _bmin;
        // result and range are non-negative, and result is possibly larger
        // than range, so the cast is safe
        if(result <= static_cast<base_result>(_range))
          return result + _min;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
