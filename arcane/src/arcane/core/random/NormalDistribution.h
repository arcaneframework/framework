// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NormalDistribution.h                                        (C) 2000-2022 */
/*                                                                           */
/* Randomiser 'NormalDistribution'.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_NORMALDISTRIBUTION_H
#define ARCANE_RANDOM_NORMALDISTRIBUTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Math.h"

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
 * $Id: NormalDistribution.h 3932 2004-08-23 08:45:03Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// deterministic polar method, uses trigonometric functions
template<class UniformRandomNumberGenerator>
class NormalDistribution
{
 public:

  typedef UniformRandomNumberGenerator base_type;

  explicit NormalDistribution(base_type & rng,Real mean = 0.0,Real sigma = 1.0)
  : _rng(rng), _mean(mean), _sigma(sigma), _valid(false)
  {
  }

  // compiler-generated copy constructor is NOT fine, need to purge cache
  NormalDistribution(const NormalDistribution& other)
  : _rng(other._rng), _mean(other._mean), _sigma(other._sigma), _valid(false)
  {
  }
  // uniform_01 cannot be assigned, neither can this class

  Real operator()()
  {
    if (!_valid) {
      _r1 = _rng();
      _r2 = _rng();
      _cached_rho = math::sqrt(-2 * math::log(1.0-_r2));
      _valid = true;
    }
    else
      _valid = false;
    // Can we have a boost::mathconst please?
    const double pi = 3.14159265358979323846;
    
    return _cached_rho * (_valid ? cos(2*pi*_r1) : sin(2*pi*_r1)) * _sigma + _mean;
  }

 private:

  Uniform01<base_type> _rng;
  const Real _mean, _sigma;
  Real _r1 = 0.0;
  Real _r2 = 0.0;
  Real _cached_rho = 0.0;
  bool _valid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::random

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
