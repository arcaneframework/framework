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
  , ptrSTC(nullptr)
  {}

  ~SimpleTableComparatorUnitTest() {}

  void setUpForClass() override;
  void setUp() override;


  void testSimple() override;
  void testFullReal() override;
  void testError() override;

  void testIncludeRow() override;
  void testIncludeColumn() override;
  void testIncludeRowColumn() override;

  void testRegexRow() override;
  void testRegexColumn() override;
  void testRegexRowColumn() override;

  void testEpsilonColumn() override;
  void testEpsilonRow() override;
  void testEpsilonRowColumn() override;

  void testCompareOneElem() override;
  void testCompareWithElem() override;


  void tearDown() override;
  void tearDownForClass() override;

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
