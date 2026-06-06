// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Uniform01.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Randomize 'Uniform01'.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_UNIFORM01_H
#define ARCANE_CORE_RANDOM_UNIFORM01_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

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
 * $Id: Uniform01.h 6199 2005-11-08 13:52:46Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

// Because it is so commonly used: uniform distribution on the real [0..1)
// range.  This allows for specializations to avoid a costly FP division
/*!
 * \brief Generates a random number in the interval [0,1[.
 *
 * \note Oct2014: reports the boost 1.55 implementation which corrects
 * a bug. In the previous version, the generator could return the value 1.0,
 * which caused the program to crash when calling NormalDistribution,
 * which uses log(1.0 - r).
 */
template <class UniformRandomNumberGenerator>
class Uniform01
{
 public:

  typedef UniformRandomNumberGenerator base_type;
  typedef Real result_type;
  static const bool has_fixed_range = false;

  explicit Uniform01(base_type& rng)
  : _rng(rng)
  {}
  // compiler-generated copy ctor is fine
  // assignment is disallowed because there is a reference member

  //! Generates a random number within the interval [0,1[.
  Real operator()()
  {
    return apply(_rng);
  }

  /*!
   * \brief Generates a random number within the interval [0,1[ from
   * the generator \a _rng.
   */
  static Real apply(base_type& _rng)
  {
    // Since _rng() can return 1.0, it loops, restarting the generator
    // as long as the returned value is 1.0. To avoid an infinite loop,
    // it performs a maximum of 100 iterations (the probability of
    // drawing the value 1.0 100 times should be almost zero).
    Real rng_val = _apply(_rng, _rng());
    for (int x = 0; x < 100; ++x) {
      if (rng_val < 1.0)
        return rng_val;
      rng_val = _apply(_rng, _rng());
    }
    return 1.0 - 1.0e-10;
  }

  /*!
   * \deprecated Use apply(base_type&) instead.
   * \note For compatibility reasons, this version contains a temporary
   * workaround to ensure that the returned value is different from 1.0.
   * However, this workaround is not necessarily very relevant statistically.
   */
  static ARCANE_DEPRECATED_122 Real apply(const base_type& _rng, typename base_type::result_type rng_val)
  {
    Real r = _apply(_rng, rng_val);
    if (r < 1.0)
      return r;
    // Returns a number close to 1 but less than it.
    return (1.0 - 1.0e-10);
  }
  //! Minimum returned value.
  static Real min() { return 0.0; }
  //! Upper bound (not reached) of the returned values.
  static Real max() { return 1.0; }

 private:

  static Real _apply(const base_type& _rng, typename base_type::result_type rng_val)
  {
    return static_cast<Real>(rng_val - _rng.min()) /
    (static_cast<Real>(_rng.max() - _rng.min()) +
     (std::numeric_limits<base_result>::is_integer ? 1.0 : 0.0));
  }

 private:

  typedef typename base_type::result_type base_result;
  base_type& _rng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::random

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
