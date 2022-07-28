// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorUnitTest.hh                            (C) 2000-2022 */
/*                                                                           */
/* Service de test pour les services implémentant ISimpleTableComparator.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicUnitTest.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISimpleTableComparator.h"
#include "arcane/ISimpleTableOutput.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ServiceBuilder.h"

#include "arcane/tests/SimpleTableComparatorUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleTableComparatorUnitTest
: public ArcaneSimpleTableComparatorUnitTestObject
{
 public:
  SimpleTableComparatorUnitTest(const ServiceBuildInfo& sbi)
  : ArcaneSimpleTableComparatorUnitTestObject(sbi)
  , ptrSTO(nullptr)
  {}

  ~SimpleTableComparatorUnitTest() {}

  void setUpForClass() override;
  void setUp() override;

  void testInit() override;

  void tearDown() override;
  void tearDownForClass() override;

 private:
  template <class T>
  void ASSERT_EQUAL_ARRAY(UniqueArray<T> expected, UniqueArray<T> actual);

 private:
  ISimpleTableOutput* ptrSTO;
  ISimpleTableComparator* ptrSTC;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLETABLECOMPARATORUNITTEST(SimpleTableComparatorUnitTest, SimpleTableComparatorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
