// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RandomNumberGeneratorUnitTest.h                             (C) 2000-2025 */
/*                                                                           */
/* Service de test des générateurs de nombres (pseudo-)aléatoires            */
/* implémentant l'interface IRandomNumberGenerator                           */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_TESTS_RANDOMNUMBERGENERATORUNITTEST_H
#define ARCANE_TESTS_RANDOMNUMBERGENERATORUNITTEST_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IRandomNumberGenerator.h"
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

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
