// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorService.cc                         (C) 2000-2022 */
/*                                                                           */
/* Implémentation d'un générateur de nombres aléatoires LCG.                 */
/* Inspiré du générateur de Quicksilver (LLNL) et des pages 302-304          */
/* du livre :                                                                */
/*                                                                           */
/*   Numerical Recipes in C                                                  */
/*   The Art of Scientific Computing                                         */
/*   Second Edition                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/PDESRandomNumberGeneratorService.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorService::
initSeed()
{
  if (m_with_option) {
    m_seed = options()->getInitialSeed();
  }
  else {
    m_seed = 4294967297;
  }
}

void PDESRandomNumberGeneratorService::
initSeed(RandomNumberGeneratorSeed seed)
{
  if (seed.sizeOfSeed() != sizeof(Int64)) {
    ARCANE_FATAL("Bad size of seed");
  }
  seed.seed(m_seed);
}

RandomNumberGeneratorSeed PDESRandomNumberGeneratorService::
seed()
{
  return RandomNumberGeneratorSeed(m_seed);
}

// Les sauts négatifs sont supportés.
RandomNumberGeneratorSeed PDESRandomNumberGeneratorService::
generateRandomSeed(Integer leap)
{
  // Pas besoin de faire de saut si leap == 0.
  if (leap != 0) {
    _ran4(&m_seed, leap - 1);
  }
  Int64 spawned_seed = _hashState(m_seed);
  _ran4(&m_seed, 0);
  return RandomNumberGeneratorSeed(spawned_seed);
}

// Les sauts négatifs sont supportés.
RandomNumberGeneratorSeed PDESRandomNumberGeneratorService::
generateRandomSeed(RandomNumberGeneratorSeed* parent_seed, Integer leap)
{
  Int64 i_seed;
  parent_seed->seed(i_seed);

  // Pas besoin de faire de saut si leap == 0.
  if (leap != 0) {
    _ran4(&i_seed, leap - 1);
  }
  Int64 spawned_seed = _hashState(i_seed);
  _ran4(&i_seed, 0);
  parent_seed->setSeed(i_seed);
  return RandomNumberGeneratorSeed(spawned_seed);
}

// Les sauts négatifs sont supportés.
Real PDESRandomNumberGeneratorService::
generateRandomNumber(Integer leap)
{
  return _ran4(&m_seed, leap);
}

// Les sauts négatifs sont supportés.
Real PDESRandomNumberGeneratorService::
generateRandomNumber(RandomNumberGeneratorSeed* seed, Integer leap)
{
  Int64 i_seed;
  seed->seed(i_seed);
  Real fin = _ran4(&i_seed, leap);
  seed->setSeed(i_seed);
  return fin;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Méthode permettant de découper un uint64 en deux uint32.
 * 
 * @param uint64_in Le uint64 à découper en deux.
 * @param front_bits Les 32 bits de poids fort.
 * @param back_bits Les 32 bits de poids faible.
 */
void PDESRandomNumberGeneratorService::
_breakupUInt64(uint64_t uint64_in, uint32_t* front_bits, uint32_t* back_bits)
{
  *front_bits = static_cast<uint32_t>(uint64_in >> 32);
  *back_bits = static_cast<uint32_t>(uint64_in & 0xffffffff);
}

/**
 * @brief Méthode permettant de regrouper deux uint32 en un uint64.
 * 
 * @param front_bits Les 32 bits de poids fort.
 * @param back_bits Les 32 bits de poids faible.
 * @return uint64_t Le uint64 reconstitué.
 */
uint64_t PDESRandomNumberGeneratorService::
_reconstructUInt64(uint32_t front_bits, uint32_t back_bits)
{
  uint64_t reconstructed, temp;
  reconstructed = static_cast<uint64_t>(front_bits);
  temp = static_cast<uint64_t>(back_bits);

  // shift first bits 32 bits to left
  reconstructed = reconstructed << 32;

  // temp must be masked to kill leading 1's.  Then 'or' with reconstructed
  // to get the last bits in
  reconstructed |= (temp & 0x00000000ffffffff);

  return reconstructed;
}

/**
 * @brief Algorithme Pseudo-DES du livre :
 * Numerical Recipes in C
 * The Art of Scientific Computing
 * Second Edition
 * 
 * (Pages 302-303)
 * 
 * @param lword Moitié de gauche.
 * @param irword Moitié de droite.
 */
void PDESRandomNumberGeneratorService::
_psdes(uint32_t* lword, uint32_t* irword)
{
  const Integer NITER = 4;
  const uint32_t c1[] = { 0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L };
  const uint32_t c2[] = { 0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L };

  for (Integer i = 0; i < NITER; i++) {
    uint32_t iswap = iswap = (*irword);
    uint32_t ia = iswap ^ c1[i];
    uint32_t itmpl = ia & 0xffff;
    uint32_t itmph = ia >> 16;
    uint32_t ib = itmpl * itmpl + ~(itmph * itmph);

    *irword = (*lword) ^ (((ia = (ib >> 16) | ((ib & 0xffff) << 16)) ^ c2[i]) + itmpl * itmph);
    *lword = iswap;
  }
}

/**
 * @brief Méthode permettant de générer une nouvelle graine avec l'algorithme
 * pseudo-DES.
 * 
 * @param initial_number La graine "parent".
 * @return uint64_t La graine "enfant".
 */
uint64_t PDESRandomNumberGeneratorService::
_hashState(uint64_t initial_number)
{
  uint32_t front_bits, back_bits;
  _breakupUInt64(initial_number, &front_bits, &back_bits);

  _psdes(&front_bits, &back_bits);

  uint64_t fin = _reconstructUInt64(front_bits, back_bits);
  return fin;
}

/**
 * @brief Méthode permettant de générer des nombres pseudo-aléatoire
 * à partir d'une graine.
 * 
 * Inspiré de l'algorithme ran4 du livre :
 * Numerical Recipes in C
 * The Art of Scientific Computing
 * Second Edition
 * 
 * (Pages 303-304)
 * 
 * @param seed La graine.
 * @param leap Le saut.
 */
Real PDESRandomNumberGeneratorService::
_ran4(Int64* seed, Integer leap)
{
  uint32_t front_bits, back_bits, irword, lword;
  _breakupUInt64((uint64_t)(*seed), &front_bits, &back_bits);
  front_bits += leap;

  irword = front_bits;
  lword = back_bits;
  _psdes(&lword, &irword);

  *seed = (Int64)_reconstructUInt64(++front_bits, back_bits);

  volatile Real fin = 5.4210108624275222e-20 * (Real)_reconstructUInt64(lword, irword);
  return fin;
}