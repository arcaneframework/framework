// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputUnitTest.h                                 (C) 2000-2025 */
/*                                                                           */
/* Service de test pour les services implémentant ISimpleTableOutput.        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_TESTS_SIMPLETABLEOUTPUTUNITTEST_H
#define ARCANE_TESTS_SIMPLETABLEOUTPUTUNITTEST_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ISimpleTableOutput.h"
#include "arcane/core/ITimeLoopMng.h"

#include "arcane/tests/SimpleTableOutputUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleTableOutputUnitTest
: public ArcaneSimpleTableOutputUnitTestObject
{
 public:

  SimpleTableOutputUnitTest(const ServiceBuildInfo& sbi)
  : ArcaneSimpleTableOutputUnitTestObject(sbi)
  , ptrSTO(nullptr)
  {}

  ~SimpleTableOutputUnitTest() {}

  void setUpForClass() override;
  void setUp() override;

  void testInit() override;

  void testAddRow1() override;
  void testAddRow2() override;
  void testAddRow3() override;
  void testAddRows1() override;

  void testAddColumn1() override;
  void testAddColumn2() override;
  void testAddColumn3() override;
  void testAddColumn4() override;
  void testAddColumns1() override;

  void testAddElemRow1() override;
  void testAddElemRow2() override;
  void testAddElemSameRow1() override;

  void testAddElemsRow1() override;
  void testAddElemsRow2() override;
  void testAddElemsSameRow1() override;

  void testAddElemColumn1() override;
  void testAddElemColumn2() override;
  void testAddElemSameColumn1() override;

  void testAddElemsColumn1() override;
  void testAddElemsColumn2() override;
  void testAddElemsSameColumn1() override;

  void testAddElemSame1() override;

  void testEditElem1() override;
  void testEditElem2() override;

  void testElem1() override;
  void testElem2() override;

  void testSizeRow1() override;
  void testSizeRow2() override;

  void testSizeColumn1() override;
  void testSizeColumn2() override;

  void testPosRowColumn1() override;

  void testNumRowColumn1() override;

  void testAddRowSameColumn1() override;
  void testAddRowSameColumn2() override;
  void testAddRowSameColumn3() override;

  void testAddColumnSameRow1() override;
  void testAddColumnSameRow2() override;
  void testAddColumnSameRow3() override;

  void testEditElemUDLR1() override;

  void testEditElemDown1() override;
  void testEditElemRight1() override;

  void testEditNameRow() override;
  void testEditNameColumn() override;

  void testWriteFile() override;

  void tearDown() override;
  void tearDownForClass() override;

 private:

  template <class T>
  void _assertEqualArray(const UniqueArray<T>& expected, const UniqueArray<T>& actual);

 private:

  ISimpleTableOutput* ptrSTO;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
