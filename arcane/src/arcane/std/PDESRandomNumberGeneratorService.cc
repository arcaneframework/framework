// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorService.cc                         (C) 2000-2022 */
/*                                                                           */
/* Implementation of an LCG random number generator.                         */
/* Inspired by the Quicksilver generator (LLNL) and pages 302-304            */
/* of the book:                                                              */
/*                                                                           */
/*   Numerical Recipes in C                                                  */
/*   The Art of Scientific Computing                                         */
/*   Second Edition                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/PDESRandomNumberGeneratorService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PDESRandomNumberGeneratorService::
initSeed()
{
  if (m_with_option) {
    m_seed = options()->getInitialSeed();
  }
  else {
    m_seed = 4294967297;
  }
  return true;
}

bool PDESRandomNumberGeneratorService::
initSeed(ByteArrayView seed)
{
  RNGSeedHelper helper(seed);
  if (helper.sizeOfSeed() != m_size_of_seed) {
    return false;
  }
  helper.value(m_seed);
  return true;
}

ByteConstArrayView PDESRandomNumberGeneratorService::
viewSeed()
{
  return RNGSeedHelper(&m_seed).constView();
}

ByteUniqueArray PDESRandomNumberGeneratorService::
emptySeed()
{
  return ByteUniqueArray(m_size_of_seed);
}

Integer PDESRandomNumberGeneratorService::
neededSizeOfSeed()
{
  return m_size_of_seed;
}

// Negative leaps are supported.
ByteUniqueArray PDESRandomNumberGeneratorService::
generateRandomSeed(Integer leap)
{
  // No need to leap if leap == 0.
  if (leap != 0) {
    _ran4(&m_seed, leap - 1);
  }
  Int64 spawned_seed = _hashState(m_seed);
  _ran4(&m_seed, 0);

  return RNGSeedHelper(&spawned_seed).copy();
}

// Negative leaps are supported.
ByteUniqueArray PDESRandomNumberGeneratorService::
generateRandomSeed(ByteArrayView parent_seed, Integer leap)
{
  if (parent_seed.size() != m_size_of_seed) {
    ARCANE_FATAL("Erreur de taille de graine.");
  }
  Int64* i_seed = (Int64*)parent_seed.data();

  // No need to leap if leap == 0.
  if (leap != 0) {
    _ran4(i_seed, leap - 1);
  }
  Int64 spawned_seed = _hashState(*i_seed);
  _ran4(i_seed, 0);
  return RNGSeedHelper(&spawned_seed).copy();
}

// Negative leaps are supported.
Real PDESRandomNumberGeneratorService::
generateRandomNumber(Integer leap)
{
  return _ran4(&m_seed, leap);
}

// Negative leaps are supported.
Real PDESRandomNumberGeneratorService::
generateRandomNumber(ByteArrayView seed, Integer leap)
{
  if (seed.size() != m_size_of_seed) {
    ARCANE_FATAL("Erreur de taille de graine.");
  }
  Int64* i_seed = (Int64*)seed.data();
  Real fin = _ran4(i_seed, leap);
  return fin;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Method to split a uint64 into two uint32s.
 * 
 * @param uint64_in The uint64 to split into two.
 * @param front_bits The 32 high-order bits.
 * @param back_bits The 32 low-order bits.
 */
void PDESRandomNumberGeneratorService::
_breakupUInt64(uint64_t uint64_in, uint32_t* front_bits, uint32_t* back_bits)
{
  *front_bits = static_cast<uint32_t>(uint64_in >> 32);
  *back_bits = static_cast<uint32_t>(uint64_in & 0xffffffff);
}

/**
 * @brief Method to combine two uint32s into a uint64.
 * 
 * @param front_bits The 32 high-order bits.
 * @param back_bits The 32 low-order bits.
 * @return uint64_t The reconstructed uint64.
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
 * @brief Pseudo-DES algorithm from the book:
 * Numerical Recipes in C
 * The Art of Scientific Computing
 * Second Edition
 * 
 * (Pages 302-303)
 * 
 * @param lword Left half.
 * @param irword Right half.
 */
void PDESRandomNumberGeneratorService::
_psdes(uint32_t* lword, uint32_t* irword)
{
  const Integer NITER = 4;
  const uint32_t c1[] = { 0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L };
  const uint32_t c2[] = { 0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L };

  for (Integer i = 0; i < NITER; i++) {
    uint32_t iswap = (*irword);
    uint32_t ia = iswap ^ c1[i];
    uint32_t itmpl = ia & 0xffff;
    uint32_t itmph = ia >> 16;
    uint32_t ib = itmpl * itmpl + ~(itmph * itmph);

    *irword = (*lword) ^ (((ia = (ib >> 16) | ((ib & 0xffff) << 16)) ^ c2[i]) + itmpl * itmph);
    *lword = iswap;
  }
}

/**
 * @brief Method to generate a new seed using the pseudo-DES algorithm.
 * 
 * @param initial_number The "parent" seed.
 * @return uint64_t The "child" seed.
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
 * @brief Method to generate pseudo-random numbers from a seed.
 * 
 * Inspired by the ran4 algorithm from the book:
 * Numerical Recipes in C
 * The Art of Scientific Computing
 * Second Edition
 * 
 * (Pages 303-304)
 * 
 * @param seed The seed.
 * @param leap The leap.
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
