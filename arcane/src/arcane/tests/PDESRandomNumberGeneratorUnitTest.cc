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

#include "arcane/BasicUnitTest.h"
#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/IRandomNumberGenerator.h"

#include "arcane/tests/PDESRandomNumberGeneratorUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PDESRandomNumberGeneratorUnitTest
: public ArcanePDESRandomNumberGeneratorUnitTestObject
{

public:

  PDESRandomNumberGeneratorUnitTest(const ServiceBuildInfo& sb)
    : ArcanePDESRandomNumberGeneratorUnitTestObject(sb) {}
  
  ~PDESRandomNumberGeneratorUnitTest() {}

 public:
  void initializeTest() override;
  void executeTest() override;

 private:
  Real testSameNumber1();
  Real testSameNumber2(Int64* seed);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PDESRANDOMNUMBERGENERATORUNITTEST(PDESRandomNumberGeneratorUnitTest,PDESRandomNumberGeneratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorUnitTest::
initializeTest()
{ 
  info() << "init test";
  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();
  rng->initSeed(1234);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PDESRandomNumberGeneratorUnitTest::
executeTest()
{
  info() << "execute test";

  RealUniqueArray hardcoded_val{
    0.516725, 0.951576, 0.214017, 0.113452, 0.136498, 0.297928, 0.848614,
    0.693764, 0.0341259, 0.335914, 0.52437, 0.076249, 0.195428, 0.385534,
    0.808555, 0.790372, 0.513851, 0.228777, 0.940043, 0.95412, 0.974406,
    0.454503, 0.982343, 0.182935, 0.414793, 0.717564, 0.721307, 0.538425,
    0.769439, 0.338561, 0.252165, 0.781353, 0.243085, 0.897954, 0.630723,
    0.203992, 0.892035, 0.410771, 0.835602, 0.984837, 0.467565, 0.115473,
    0.57945, 0.36899, 0.294814, 0.51338, 0.912344, 0.883036, 0.422084,
    0.00667469, 0.225418, 0.944801, 0.0627036, 0.158594, 0.172918, 0.275475,
    0.663461, 0.0480546, 0.749866, 0.920464, 0.350292, 0.405152, 0.122304,
    0.242838, 0.256839, 0.597857, 0.833907, 0.262299, 0.823824, 0.821539,
    0.789311, 0.325754, 0.0626824, 0.958881, 0.753192, 0.062305, 0.774736,
    0.186629, 0.826519, 0.706924, 0.294629, 0.78588, 0.652581, 0.584414,
    0.0227777, 0.56788, 0.831793, 0.532689, 0.14504, 0.327888, 0.0777679,
    0.722275, 0.785179, 0.889842, 0.297335, 0.746968, 0.607782, 0.831934,
    0.688667, 0.945306 };

  IRandomNumberGenerator* rng = options()->getPDESRandomNumberGenerator();
  Int64 initial_seed = 1234;
  for(Integer i = 0; i < 100; i++){
    Real val1 = rng->generateRandomNumber();
    Real val2 = rng->generateRandomNumber(&initial_seed);
    if(val1 != val2) ARCANE_FATAL("[1] Valeurs differentes.");
    if((Integer)(hardcoded_val[i]*10000) != (Integer)(val1*10000)) ARCANE_FATAL("[2] Valeurs differentes.");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real PDESRandomNumberGeneratorUnitTest::
testSameNumber1()
{

}

Real PDESRandomNumberGeneratorUnitTest::
testSameNumber2(Int64* seed)
{

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
