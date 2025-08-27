// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Uniform01.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Randomiser 'Uniform01'.                                                   */
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
 * \brief Génère un nombre aléatoire dans l'intervalle [0,1[.
 *
 * \note Oct2014: reporte l'implémentation de boost 1.55 qui corrige
 * un bug. Dans la version d'avant, le générateur pouvait retourner
 * la valeur 1.0 ce qui faisait planter le programme lors de
 * l'appel à NormalDistribution qui utilise log(1.0 - r).
 */
template<class UniformRandomNumberGenerator>
class Uniform01
{
 public:
  typedef UniformRandomNumberGenerator base_type;
  typedef Real result_type;
  static const bool has_fixed_range = false;

  explicit Uniform01(base_type & rng)
  : _rng(rng) { }
  // compiler-generated copy ctor is fine
  // assignment is disallowed because there is a reference member

  //! Génère un nombre aléatoire compris dans l'intervalle [0,1[.
  Real operator()()
  {
    return apply(_rng);
  }

  /*!
   * \brief Génère un nombre aléatoire compris dans l'intervalle [0,1[ à partir
   * du générateur \a _rng.
   */
  static Real apply(base_type& _rng)
  {
    // Comme _rng() peut retourner 1.0, fait une boucle
    // permettant de relancer le générateur tant que
    // la valeur retournée vaut 1.0. Pour éviter
    // une boucle infinie, effectue au maximum 100 itérations
    // (la probabilité de tirer 100 fois la valeur 1.0
    // devrait être quasiment nulle).
    Real rng_val = _apply(_rng,_rng());
    for(int x=0;x<100;++x){
      if (rng_val<1.0)
        return rng_val;
      rng_val = _apply(_rng,_rng());
    }
    return 1.0-1.0e-10;
  }

  /*!
   * \deprecated Utiliser apply(base_type&) à la place.
   * \note Pour des raisons de compatibilité, cette version contient
   * un contournement temporaire permettant de garantir
   * que la valeur retournée est différente de 1.0. Cependant,
   * ce contournement n'est pas forcément très pertinent au niveau
   * de la statistique.
   */
  static ARCANE_DEPRECATED_122 Real apply(const base_type& _rng,typename base_type::result_type rng_val)
  {
    Real r = _apply(_rng,rng_val);
    if (r<1.0)
      return r;
    // Retourne un nombre proche de 1 mais inférieur.
    return (1.0-1.0e-10);
  }
  //! Valeur minimal retournée.
  static Real min() { return 0.0; }
  //! Borne supérieure (non atteinte) des valeurs retournées.
  static Real max() { return 1.0; }

 private:

  static Real _apply(const base_type& _rng,typename base_type::result_type rng_val)
  {
    return static_cast<Real>(rng_val - _rng.min()) /
      (static_cast<Real>(_rng.max()-_rng.min()) +
       (std::numeric_limits<base_result>::is_integer ? 1.0 : 0.0));
  }

 private:

  typedef typename base_type::result_type base_result;
  base_type & _rng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
