// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorUnitTest.cc                            (C) 2000-2022 */
/*                                                                           */
/* Service de test pour les services implémentant ISimpleTableComparator.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/SimpleTableComparatorUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void SimpleTableComparatorUnitTest::
ASSERT_EQUAL_ARRAY(UniqueArray<T> expected, UniqueArray<T> actual)
{
  ASSERT_EQUAL(expected.size(), actual.size());
  for (Integer i = 0; i < actual.size(); i++) {
    ASSERT_EQUAL(expected[i], actual[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableComparatorUnitTest::
setUpForClass()
{
  ptrSTO = options()->getSimpleTableOutput();
  ptrSTC = options()->getSimpleTableComparator();

  // Init STO
  ptrSTO->init("test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  RealUniqueArray result1 = { 1, 2, 2 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 8, 9 };

  ptrSTO->writeFile("test_csv_comparator", 0);

  // Init STC
  ptrSTC->addSimpleTableOutputEntry(ptrSTO);
  ptrSTC->readSimpleTableOutputEntry();

}

void SimpleTableComparatorUnitTest::
setUp()
{

}

void SimpleTableComparatorUnitTest::
testInit()
{

}

void SimpleTableComparatorUnitTest::
tearDown()
{
  // N'est pas exécuté après un test qui a échoué.
}

void SimpleTableComparatorUnitTest::
tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
