// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorUnitTest.cc                        (C) 2000-2022 */
/*                                                                           */
/* Service de test du générateur de nombres (pseudo-)aléatoires avec         */
/* l'algorithme pseudo-DES.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/PDESRandomNumberGeneratorUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void PDESRandomNumberGeneratorUnitTest::
setUpForClass()
{
  ptrRNG = options()->getPdesRandomNumberGenerator();
}

void PDESRandomNumberGeneratorUnitTest::
setUp()
{
}

void PDESRandomNumberGeneratorUnitTest::
testHardcodedValues()
{
  RandomNumberGeneratorSeed r_seed = (ptrRNG->emptySeed() = hardcoded_seed);
  ptrRNG->initSeed(r_seed);
  RandomNumberGeneratorSeed initial_seed(r_seed);

  for (Integer i = 0; i < hardcoded_vals.size(); i++) {
    Real val1 = ptrRNG->generateRandomNumber();
    Real val2 = ptrRNG->generateRandomNumber(&initial_seed);

    ASSERT_EQUAL(val1, val2);
    ASSERT_NEARLY_EQUAL(val1, hardcoded_vals[i]);
  }
}

void PDESRandomNumberGeneratorUnitTest::
testHardcodedSeeds()
{
  RandomNumberGeneratorSeed r_seed = (ptrRNG->emptySeed() = hardcoded_seed);
  ptrRNG->initSeed(r_seed);
  RandomNumberGeneratorSeed initial_seed(r_seed);

  for (Integer i = 0; i < hardcoded_seeds.size(); i++) {
    RandomNumberGeneratorSeed val11 = ptrRNG->generateRandomSeed();
    RandomNumberGeneratorSeed val22 = ptrRNG->generateRandomSeed(&initial_seed);

    // On peut mettre direct Int64 vu que l'on teste l'implem PDESRNGS.
    Int64 val1, val2;
    ASSERT_TRUE(val11.seed(val1, false));
    ASSERT_TRUE(val22.seed(val2, false));

    ASSERT_EQUAL(val1, val2);
    ASSERT_EQUAL(val1, hardcoded_seeds[i]);
  }
}

void PDESRandomNumberGeneratorUnitTest::
tearDown()
{
  // N'est pas exécuté après un test qui a échoué.
}

void PDESRandomNumberGeneratorUnitTest::
tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
