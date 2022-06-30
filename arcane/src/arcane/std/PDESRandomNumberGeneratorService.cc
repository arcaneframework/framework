// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorService.cc                         (C) 2000-2022 */
/*                                                                           */
/* Implémentation d'un générateur de nombres aléatoires.                     */
/* Basé sur le générateur de Quicksilver (LLNL).                             */
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
    m_seed = 1029384756;
  }
}

void PDESRandomNumberGeneratorService::
initSeed(Int64 seed)
{
  m_seed = seed;
}

Int64 PDESRandomNumberGeneratorService::
seed()
{
  return m_seed;
}

// This routine spawns a "child" random number seed from a "parent" random
// number seed.
Int64 PDESRandomNumberGeneratorService::
generateRandomSeed()
{
  Int64 spawned_seed = _hashState(m_seed);
  // Bump the parent seed as that is what is expected from the interface.
  generateRandomNumber();
  return spawned_seed;
}

Int64 PDESRandomNumberGeneratorService::
generateRandomSeed(Int64* parent_seed)
{
  Int64 spawned_seed = _hashState(*parent_seed);
  // Bump the parent seed as that is what is expected from the interface.
  generateRandomNumber(parent_seed);
  return spawned_seed;
}

// Sample returns the pseudo-random number produced by a call to a random
// number generator.
Real PDESRandomNumberGeneratorService::
generateRandomNumber()
{
  // Reset the state from the previous value.
  m_seed = 2862933555777941757ULL * (uint64_t)(m_seed) + 3037000493ULL;
  // Map the int state in (0,2**64) to double (0,1)
  // by multiplying by
  // 1/(2**64 - 1) = 1/18446744073709551615.
  volatile Real fin = 5.4210108624275222e-20 * (uint64_t)(m_seed);
  return fin;
}

Real PDESRandomNumberGeneratorService::
generateRandomNumber(Int64* seed)
{
  // Reset the state from the previous value.
  *seed = 2862933555777941757ULL * (uint64_t)(*seed) + 3037000493ULL;
  // Map the int state in (0,2**64) to double (0,1)
  // by multiplying by
  // 1/(2**64 - 1) = 1/18446744073709551615.
  volatile Real fin = 5.4210108624275222e-20 * (uint64_t)(*seed);
  return fin;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Break a 64 bit state into 2 32 bit ints.
void PDESRandomNumberGeneratorService::
_breakupUInt64(uint64_t uint64_in, uint32_t& front_bits, uint32_t& back_bits)
{
  front_bits = static_cast<uint32_t>(uint64_in >> 32);
  back_bits = static_cast<uint32_t>(uint64_in & 0xffffffff);
}

// Function used to reconstruct  a 64 bit from 2 32 bit ints.
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

// Function sed to hash a 64 bit int into another, unrelated one.  It
// does this in two 32 bit chuncks. This function uses the algorithm
// from Numerical Recipies in C, 2nd edition: psdes, p. 302.  This is
// used to make 64 bit numbers for use as initial states for the 64
// bit lcg random number generator.
void PDESRandomNumberGeneratorService::
_pseudoDES(uint32_t& lword, uint32_t& irword)
{
  // This random number generator assumes that type uint32_t is a 32 bit int
  // = 1/2 of a 64 bit int. The sizeof operator returns the size in bytes = 8
  // bits.

  const int NITER = 2;
  const uint32_t c1[] = { 0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L };
  const uint32_t c2[] = { 0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L };

  uint32_t ia, ib, iswap, itmph = 0, itmpl = 0;

  for (int i = 0; i < NITER; i++) {
    ia = (iswap = irword) ^ c1[i];
    itmpl = ia & 0xffff;
    itmph = ia >> 16;
    ib = itmpl * itmpl + ~(itmph * itmph);

    irword = lword ^ (((ia = (ib >> 16) | ((ib & 0xffff) << 16)) ^ c2[i]) + itmpl * itmph);

    lword = iswap;
  }
}

// Function used to hash a 64 bit int to get an initial state.
uint64_t PDESRandomNumberGeneratorService::
_hashState(uint64_t initial_number)
{
  // break initial number apart into 2 32 bit ints
  uint32_t front_bits, back_bits;
  _breakupUInt64(initial_number, front_bits, back_bits);

  // hash the bits
  _pseudoDES(front_bits, back_bits);

  // put the hashed parts together into 1 64 bit int
  uint64_t fin = _reconstructUInt64(front_bits, back_bits);
  return fin;
}
