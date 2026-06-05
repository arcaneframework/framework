// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorUnitTest.cc                        (C) 2000-2025 */
/*                                                                           */
/* Test service for the (pseudo-)random number generator using               */
/* the pseudo-DES algorithm.                                                 */
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
  ByteUniqueArray r_seed(ptrRNG->neededSizeOfSeed());
  RNGSeedHelper(r_seed).setValue(hardcoded_seed);
  ptrRNG->initSeed(r_seed);

  ByteUniqueArray initial_seed(r_seed);

  for (Integer i = 0; i < hardcoded_vals.size(); i++) {
    Real val1 = ptrRNG->generateRandomNumber();
    Real val2 = ptrRNG->generateRandomNumber(initial_seed);

    ASSERT_EQUAL(val1, val2);
    ASSERT_NEARLY_EQUAL(val1, hardcoded_vals[i]);
  }
}

void PDESRandomNumberGeneratorUnitTest::
testHardcodedSeeds()
{
  ByteUniqueArray r_seed = ptrRNG->emptySeed();
  RNGSeedHelper(r_seed).setValue(hardcoded_seed);
  ptrRNG->initSeed(r_seed);

  ByteUniqueArray initial_seed(r_seed);

  for (Integer i = 0; i < hardcoded_seeds.size(); i++) {
    ByteUniqueArray val11 = ptrRNG->generateRandomSeed();
    ByteUniqueArray val22 = ptrRNG->generateRandomSeed(initial_seed);

    // We can use Int64 directly since we are testing the PDESRNGS implementation.
    Int64 val1, val2;
    ASSERT_TRUE(RNGSeedHelper(val11).value(val1, false));
    ASSERT_TRUE(RNGSeedHelper(val22).value(val2, false));

    ASSERT_EQUAL(val1, val2);
    ASSERT_EQUAL(val1, hardcoded_seeds[i]);
  }
}

void PDESRandomNumberGeneratorUnitTest::
tearDown()
{
  // Is not executed after a failed test.
}

void PDESRandomNumberGeneratorUnitTest::
tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PDESRANDOMNUMBERGENERATORUNITTEST(PDESRandomNumberGeneratorUnitTest, PDESRandomNumberGeneratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
