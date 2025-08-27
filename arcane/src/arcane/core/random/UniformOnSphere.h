// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UniformOnSphere.h                                           (C) 2000-2014 */
/*                                                                           */
/* Randomiser 'Uniform0nSphere'.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_RANDOM_UNIFORMONSPHERE_H
#define ARCANE_RANDOM_UNIFORMONSPHERE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/random/NormalDistribution.h"

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
 * $Id: UniformOnSphere.h 7934 2007-01-16 08:49:47Z grospelx $
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class UniformRandomNumberGenerator>
class UniformOnSphere
{
 public:
  typedef UniformRandomNumberGenerator base_type;
  typedef Real3 result_type;

  explicit UniformOnSphere(base_type & rng)
    : _rng(rng) { }

  typedef typename base_type::result_type base_result;
  Real3 applyDim3()
    {
      Real results[3];
      Real sqsum = 0;
      for( Integer i=0; i<3; ++i ){
        Real val = _rng();
        val += 1.0e-16;
        results[i] = val;
	//cout << " VAL " << i << " =" << val << '\n';
        sqsum += val * val;
      }
      //cout << " SQSUM " << sqsum << '\n';
      Real inv_sqrt_sqsum = 1. / math::sqrt(sqsum);
      //cout << " INV_SQSUM " << inv_sqrt_sqsum << '\n';
      for( Integer i=0; i<3; ++i ){
        results[i] *= inv_sqrt_sqsum;
      }
      return Real3(results[0],results[1],results[2]);
    }
 private:

  NormalDistribution<base_type> _rng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RANDOM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
