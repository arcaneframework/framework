// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomNumberGeneratorUnitTest.cc                            (C) 2000-2022 */
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
testRNGS()
{
  Integer aaa = 1234;
  RandomNumberGeneratorSeed test_a(aaa, sizeof(Integer));

  Integer bbb;
  test_a.seed(bbb);
  ASSERT_TRUE(aaa == bbb);

  RandomNumberGeneratorSeed test_b(bbb, sizeof(Integer)); //1234
  ASSERT_TRUE(test_a == test_b);

  test_b = 1234;
  ASSERT_TRUE(test_a == test_b);

  RandomNumberGeneratorSeed test_c = test_a;
  ASSERT_TRUE(test_a == test_c);

  test_c.resize(sizeof(Int64));
  ASSERT_TRUE(test_a != test_c);

  test_b = (ptrRNG->emptySeed() = 1234);
  ASSERT_TRUE(test_b == test_c);
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
  if(!ptrRNG->isLeapNumberSupported()) return;

  RandomNumberGeneratorSeed r_seed = ptrRNG->emptySeed();
  r_seed = 1234;
  ptrRNG->initSeed(r_seed);

  RealUniqueArray result(100);

  for (Integer i = 0; i < result.size(); i++) {
    result[i] = ptrRNG->generateRandomNumber();
  }

  ptrRNG->initSeed(r_seed);

  for (Integer i = 2; i < result.size(); i+=3) {
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
  if (!ptrRNG->isLeapSeedSupported()) return;

  RandomNumberGeneratorSeed r_seed = (ptrRNG->emptySeed() = 5678);
  ptrRNG->initSeed(r_seed);

  UniqueArray<RandomNumberGeneratorSeed> result(100);

  for (Integer i = 0; i < result.size(); i++) {
    result[i] = ptrRNG->generateRandomSeed();
  }

  ptrRNG->initSeed(r_seed);

  for (Integer i = 2; i < result.size(); i += 3) {
    RandomNumberGeneratorSeed seed = ptrRNG->generateRandomSeed(2);
    ASSERT_TRUE(result[i] == seed);
  }

  // On teste aussi les sauts négatifs.
  for (Integer i = result.size() - 3; i >= 0; i--) {
    RandomNumberGeneratorSeed seed = ptrRNG->generateRandomSeed(-2);
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

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
