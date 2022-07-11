// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomNumberGeneratorUnitTest.h                             (C) 2000-2022 */
/*                                                                           */
/* Service de test des générateurs de nombres (pseudo-)aléatoires            */
/* implémentant l'interface IRandomNumberGenerator                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicUnitTest.h"
#include "arcane/IRandomNumberGenerator.h"
#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/RandomNumberGeneratorUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RandomNumberGeneratorUnitTest
: public ArcaneRandomNumberGeneratorUnitTestObject
{

 public:
  explicit RandomNumberGeneratorUnitTest(const ServiceBuildInfo& sb)
  : ArcaneRandomNumberGeneratorUnitTestObject(sb)
  , ptrRNG(nullptr)
  {}

  ~RandomNumberGeneratorUnitTest() {}

 public:
  void setUpForClass() override;
  void setUp() override;

  void testRNGS() override;
  void testMcPi() override;
  void testLeepNumbers() override;
  void testLeepSeeds() override;

  void tearDown() override;
  void tearDownForClass() override;

 private:
  IRandomNumberGenerator* ptrRNG;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_RANDOMNUMBERGENERATORUNITTEST(RandomNumberGeneratorUnitTest, RandomNumberGeneratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
