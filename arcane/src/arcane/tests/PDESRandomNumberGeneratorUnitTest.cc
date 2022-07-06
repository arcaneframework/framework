// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PDESRandomNumberGeneratorUnitTest.cc                        (C) 2000-2022 */
/*                                                                           */
/* Service du test du générateur de nombres (pseudo-)aléatoires avec         */
/* algorithme pseudo-DES.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/PDESRandomNumberGeneratorUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorUnitTest::
initializeTest()
{
  info() << "init test";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorUnitTest::
executeTest()
{
  info() << "execute test";

  
  testRNGS();
  hardcodedValues();
  hardcodedSeeds();
  mcPi();
  leepSeeds();
  leepNumbers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorUnitTest::
testRNGS()
{
  Integer aaa = 1234;
  RandomNumberGeneratorSeed test_a(aaa);

  Integer bbb;
  test_a.seed(bbb);

  if (aaa != bbb)
    ARCANE_FATAL("[testRNGS:0] Valeurs differentes.");

  RandomNumberGeneratorSeed test_b(bbb); //1234

  if (test_a != test_b)
    ARCANE_FATAL("[testRNGS:1] Valeurs differentes.");

  test_b = 1234;

  if (test_a != test_b)
    ARCANE_FATAL("[testRNGS:2] Valeurs differentes.");

  RandomNumberGeneratorSeed test_c = test_a;

  if (test_a != test_c)
    ARCANE_FATAL("[testRNGS:3] Valeurs differentes.");
}

void PDESRandomNumberGeneratorUnitTest::
hardcodedValues()
{
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();

  RandomNumberGeneratorSeed r_seed(hardcoded_seed);
  rng->initSeed(r_seed);
  RandomNumberGeneratorSeed initial_seed(hardcoded_seed);

  for (Integer i = 0; i < hardcoded_vals.size(); i++) {
    Real val1 = rng->generateRandomNumber();
    Real val2 = rng->generateRandomNumber(&initial_seed);

    if (val1 != val2) {
      info() << val1 << val2;
      ARCANE_FATAL("[hardcodedValues:0] Valeurs differentes.");
    }

    if ((Integer)(hardcoded_vals[i] * 1e9) != (Integer)(val1 * 1e9)) {
      info() << hardcoded_vals[i] << " " << val1;
      ARCANE_FATAL("[hardcodedValues:1] Valeurs differentes.");
    }
  }
}

void PDESRandomNumberGeneratorUnitTest::
hardcodedSeeds()
{
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();

  RandomNumberGeneratorSeed r_seed(hardcoded_seed);
  rng->initSeed(r_seed);
  RandomNumberGeneratorSeed initial_seed(hardcoded_seed);

  for (Integer i = 0; i < hardcoded_seeds.size(); i++) {
    RandomNumberGeneratorSeed val11 = rng->generateRandomSeed();
    RandomNumberGeneratorSeed val22 = rng->generateRandomSeed(&initial_seed);

    Int64 val1,val2;
    val11.seed(val1);
    val22.seed(val2);

    if (val1 != val2){
      info() << val1 << val2;
      ARCANE_FATAL("[hardcodedSeeds:0] Valeurs differentes.");
    }
    if (hardcoded_seeds[i] != val1){
      info() << hardcoded_seeds[i] << " " << val1;
      ARCANE_FATAL("[hardcodedSeeds:1] Valeurs differentes.");
    }
  }
}

void PDESRandomNumberGeneratorUnitTest::
mcPi()
{
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();
  rng->initSeed();

  const Integer nb_iter(10000);
  Real sum(0.);

  for (Integer i = 0; i < nb_iter; i++) {
    Real2 xy(rng->generateRandomNumber(), rng->generateRandomNumber());
    if (xy.squareNormL2() < 1)
      sum++;
  }
  Real estim = 4 * sum / nb_iter;
  info() << "Pi ~= " << estim;
  if (estim < 3.00 || estim > 3.50)
    ARCANE_FATAL("[mcPi:0] Pi.");
}

void PDESRandomNumberGeneratorUnitTest::
leepNumbers()
{
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();
  if(!rng->isLeapNumberSupported()) return;

  RandomNumberGeneratorSeed r_seed(hardcoded_seed);
  rng->initSeed(r_seed);

  for (Integer i = 2; i < hardcoded_vals.size(); i+=3) {
    Real val1 = rng->generateRandomNumber(2);

    if ((Integer)(hardcoded_vals[i] * 1e9) != (Integer)(val1 * 1e9)) {
      info() << hardcoded_vals[i] << " " << val1;
      ARCANE_FATAL("[leepNumbers:0] Valeurs differentes.");
    }
  }

  // On teste aussi les sauts négatifs.
  for (Integer i = hardcoded_vals.size() - 3; i >= 0; i--) {
    Real val1 = rng->generateRandomNumber(-2);

    if ((Integer)(hardcoded_vals[i] * 1e9) != (Integer)(val1 * 1e9)) {
      info() << hardcoded_vals[i] << " " << val1;
      ARCANE_FATAL("[leepNumbers:1] Valeurs differentes.");
    }
  }
}

void PDESRandomNumberGeneratorUnitTest::
leepSeeds()
{
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();
  if (!rng->isLeapSeedSupported())
    return;

  RandomNumberGeneratorSeed r_seed(hardcoded_seed);
  rng->initSeed(r_seed);

  for (Integer i = 2; i < hardcoded_seeds.size(); i += 3) {
    RandomNumberGeneratorSeed val11 = rng->generateRandomSeed(2);

    Int64 val1;
    val11.seed(val1);

    if (hardcoded_seeds[i] != val1) {
      info() << hardcoded_seeds[i] << " " << val1;
      ARCANE_FATAL("[leepSeeds:0] Valeurs differentes.");
    }
  }

  // On teste aussi les sauts négatifs.
  for (Integer i = hardcoded_seeds.size() - 3; i >= 0; i--) {
    RandomNumberGeneratorSeed val11 = rng->generateRandomSeed(-2);

    Int64 val1;
    val11.seed(val1);

    if (hardcoded_seeds[i] != val1) {
      info() << hardcoded_seeds[i] << " " << val1;
      ARCANE_FATAL("[leepSeeds:1] Valeurs differentes.");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
