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

#ifndef ARCANE_STD_PDESRANDOMNUMBERGENERATORSERVICE_H
#define ARCANE_STD_PDESRANDOMNUMBERGENERATORSERVICE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IRandomNumberGenerator.h"
#include "arcane/std/PDESRandomNumberGenerator_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PDESRandomNumberGeneratorService
: public ArcanePDESRandomNumberGeneratorObject
{
 public:
  PDESRandomNumberGeneratorService(const ServiceBuildInfo& sbi)
  : ArcanePDESRandomNumberGeneratorObject(sbi)
  , m_seed(4294967297)
  , m_size_of_seed(sizeof(Int64))
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~PDESRandomNumberGeneratorService(){};

 public:
  bool initSeed() override;
  bool initSeed(RandomNumberGeneratorSeed seed) override;
  bool initSeed(ByteArrayView seed) override;

  RandomNumberGeneratorSeed seed() override;
  RandomNumberGeneratorSeed emptySeed() override;
  ByteArrayView viewSeed() override;

  Integer neededSizeOfSeed() override;

  bool isLeapSeedSupported() override { return true; };
  RandomNumberGeneratorSeed generateRandomSeed(Integer leap) override;
  ByteUniqueArray generateRandomSeedBUA(Integer leap = 0) override;

  RandomNumberGeneratorSeed generateRandomSeed(RandomNumberGeneratorSeed* parent_seed, Integer leap) override;
  ByteUniqueArray generateRandomSeed(ByteArrayView parent_seed, Integer leap = 0) override;

  bool isLeapNumberSupported() override { return true; };
  Real generateRandomNumber(Integer leap) override;
  Real generateRandomNumber(RandomNumberGeneratorSeed* seed, Integer leap) override;
  Real generateRandomNumber(ByteArrayView seed, Integer leap = 0) override;

 protected:
  void _breakupUInt64(uint64_t uint64_in, uint32_t* front_bits, uint32_t* back_bits);
  uint64_t _reconstructUInt64(uint32_t front_bits, uint32_t back_bits);
  void _psdes(uint32_t* lword, uint32_t* irword);
  uint64_t _hashState(uint64_t initial_number);
  Real _ran4(Int64* seed, Integer leap);

 protected:
  Int64 m_seed;
  Integer m_size_of_seed;
  bool m_with_option;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PDESRANDOMNUMBERGENERATOR(PDESRandomNumberGenerator, PDESRandomNumberGeneratorService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
