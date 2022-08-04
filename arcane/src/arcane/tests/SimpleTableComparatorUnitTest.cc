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
}

void SimpleTableComparatorUnitTest::
setUp()
{
  ptrSTO->clear();
  ptrSTC->clear();
}

void SimpleTableComparatorUnitTest::
testSimple()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 1", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());
  ptrSTC->print();
  ptrSTC->editRegexColumns("^.*1$");
  ptrSTC->isARegexExclusiveColumns(true);

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testFullReal()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1.234567891, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 456789101112, -1056789.21354689, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, -0.0001 });

  ptrSTO->writeFile("test_csv_comparator", 0);

  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());
  ASSERT_TRUE(ptrSTC->readRefFile());
  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testError()
{
  ptrSTC->print();
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ptrSTC->print();
  ASSERT_TRUE(ptrSTC->writeRefFile());
  ptrSTC->print();

  ptrSTO->print();

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 1", 0);

  ptrSTO->print();

  ASSERT_TRUE(ptrSTC->readRefFile());
  ptrSTC->print();
  ASSERT_FALSE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testIncludeRow()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 1", 0);
  ptrSTO->editElem("Ma colonne 1", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addRowForComparing("Ma ligne 3");

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testIncludeColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 1", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addColumnForComparing("Ma colonne 3");

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testIncludeRowColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 2", "Ma ligne 1", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 2", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 3", 0);

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 2", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 2", 99);
  ptrSTO->editElem("Ma colonne 3", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addColumnForComparing("Ma colonne 1");
  ptrSTC->addColumnForComparing("Ma colonne 3");
  ptrSTC->addRowForComparing("Ma ligne 1");
  ptrSTC->addRowForComparing("Ma ligne 3");

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testRegexRow()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6});
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9});
  ptrSTO->addRow("Ma ligne 4", RealUniqueArray{ 4, 5, 7});
  ptrSTO->addRow("Ma ligne 5", RealUniqueArray{ 8, 1, 2});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 4", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 5", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addRowForComparing("Ma ligne 1");
  ptrSTC->addRowForComparing("Ma ligne 2");
  ptrSTC->addRowForComparing("Ma ligne 3");

  // 3 inclus.
  ptrSTC->editRegexRows("^.*[3-9]+$");
  ptrSTC->isARegexExclusiveRows(true);

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testRegexColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3, 4, 5 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6, 7, 8 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9, 1, 2 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 4", "Ma ligne 1", 0);
  ptrSTO->editElem("Ma colonne 5", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addColumnForComparing("Ma colonne 1");
  ptrSTC->addColumnForComparing("Ma colonne 2");
  ptrSTC->addColumnForComparing("Ma colonne 3");

  // 3 inclus.
  ptrSTC->editRegexColumns("^.*[3-9]+$");
  ptrSTC->isARegexExclusiveColumns(true);

  ASSERT_TRUE(ptrSTC->compareWithRef());
}

void SimpleTableComparatorUnitTest::
testRegexRowColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2,   3, 4,   5 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5,   6, 7,   8 });

  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8,   9, 1,   2 });
  ptrSTO->addRow("Ma ligne 4", RealUniqueArray{ 4, 5,   7, 4,   5 });

  ptrSTO->addRow("Ma ligne 5", RealUniqueArray{ 8, 1,   2, 7,   8 });


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeRefFile());

  ptrSTO->editElem("Ma colonne 1", "Ma ligne 3", 0);
  ptrSTO->editElem("Ma colonne 2", "Ma ligne 4", 0);
  ptrSTO->editElem("Ma colonne 3", "Ma ligne 3", 0);
  ptrSTO->editElem("Ma colonne 4", "Ma ligne 4", 0);
  ptrSTO->editElem("Ma colonne 5", "Ma ligne 3", 0);
  ptrSTO->editElem("Ma colonne 3", "Ma ligne 1", 0);
  ptrSTO->editElem("Ma colonne 4", "Ma ligne 2", 0);
  ptrSTO->editElem("Ma colonne 3", "Ma ligne 3", 0);
  ptrSTO->editElem("Ma colonne 4", "Ma ligne 4", 0);
  ptrSTO->editElem("Ma colonne 3", "Ma ligne 5", 0);

  ASSERT_TRUE(ptrSTC->readRefFile());

  ptrSTC->addColumnForComparing("Ma colonne 5");
  ptrSTC->editRegexColumns("^.*[1-2]+$");

  ptrSTC->editRegexRows("^.*[3-4]+$");
  ptrSTC->isARegexExclusiveRows(true);

  ASSERT_TRUE(ptrSTC->compareWithRef());
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
