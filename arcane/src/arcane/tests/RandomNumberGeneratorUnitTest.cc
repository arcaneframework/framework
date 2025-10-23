// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomNumberGeneratorUnitTest.cc                            (C) 2000-2025 */
/*                                                                           */
/* Service de test des générateurs de nombres (pseudo-)aléatoires            */
/* implémentant l'interface IRandomNumberGenerator                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/RandomNumberGeneratorUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RandomNumberGeneratorUnitTest::
setUpForClass()
{
  ptrRNG = options()->getRandomNumberGenerator();
}

void RandomNumberGeneratorUnitTest::
setUp()
{
}

void RandomNumberGeneratorUnitTest::
testMcPi()
{
  ptrRNG->initSeed();

  const Integer nb_iter(10000);
  Real sum(0.);

  for (Integer i = 0; i < nb_iter; i++) {
    Real2 xy(ptrRNG->generateRandomNumber(), ptrRNG->generateRandomNumber());
    if (xy.squareNormL2() < 1)
      sum++;
  }
  Real estim = 4 * sum / nb_iter;
  info() << "Pi ~= " << estim;
  ASSERT_TRUE(estim > 3.00 && estim < 3.50);
}

void RandomNumberGeneratorUnitTest::
testLeepNumbers()
{
  if (!ptrRNG->isLeapNumberSupported())
    return;

  ByteUniqueArray r_seed(ptrRNG->neededSizeOfSeed());
  RNGSeedHelper(r_seed).setValue(1234);
  ptrRNG->initSeed(r_seed);

  RealUniqueArray result(100);

  for (Integer i = 0; i < result.size(); i++) {
    result[i] = ptrRNG->generateRandomNumber();
  }

  ptrRNG->initSeed(r_seed);

  for (Integer i = 2; i < result.size(); i += 3) {
    Real number = ptrRNG->generateRandomNumber(2);
    ASSERT_EQUAL(result[i], number);
  }

  // On teste aussi les sauts négatifs.
  for (Integer i = result.size() - 3; i >= 0; i--) {
    Real number = ptrRNG->generateRandomNumber(-2);
    ASSERT_EQUAL(result[i], number);
  }
}

void RandomNumberGeneratorUnitTest::
testLeepSeeds()
{
  if (!ptrRNG->isLeapSeedSupported())
    return;

  ByteUniqueArray r_seed(ptrRNG->neededSizeOfSeed());
  RNGSeedHelper(r_seed).setValue(5678);
  ptrRNG->initSeed(r_seed);

  UniqueArray<ByteUniqueArray> result(100);

  for (Integer i = 0; i < result.size(); i++) {
    result[i] = ptrRNG->generateRandomSeed();
  }

  ptrRNG->initSeed(r_seed);

  for (Integer i = 2; i < result.size(); i += 3) {
    ByteUniqueArray seed = ptrRNG->generateRandomSeed(2);
    ASSERT_TRUE(result[i] == seed);
  }

  // On teste aussi les sauts négatifs.
  for (Integer i = result.size() - 3; i >= 0; i--) {
    ByteUniqueArray seed = ptrRNG->generateRandomSeed(-2);
    ASSERT_TRUE(result[i] == seed);
  }
}

void RandomNumberGeneratorUnitTest::
tearDown()
{
  // N'est pas exécuté après un test qui a échoué.
}

void RandomNumberGeneratorUnitTest::
tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_RANDOMNUMBERGENERATORUNITTEST(RandomNumberGeneratorUnitTest, RandomNumberGeneratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
