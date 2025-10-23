// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorUnitTest.cc                            (C) 2000-2025 */
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());
  ptrSTC->print();
  ptrSTC->editRegexColumns("^.*1$");
  ptrSTC->isARegexExclusiveColumns(true);

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());
  ASSERT_TRUE(ptrSTC->readReferenceFile());
  ASSERT_TRUE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testError()
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());
  ASSERT_FALSE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 0);
  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addRowForComparing("Ma ligne 3");

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addColumnForComparing("Ma colonne 3");

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 2", "Ma ligne 1", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 3", 0);

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 99);
  ptrSTO->editElement("Ma colonne 3", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addColumnForComparing("Ma colonne 1");
  ptrSTC->addColumnForComparing("Ma colonne 3");
  ptrSTC->addRowForComparing("Ma ligne 1");
  ptrSTC->addRowForComparing("Ma ligne 3");

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 4", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 5", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addRowForComparing("Ma ligne 1");
  ptrSTC->addRowForComparing("Ma ligne 2");
  ptrSTC->addRowForComparing("Ma ligne 3");

  // 3 inclus.
  ptrSTC->editRegexRows("^.*[3-9]+$");
  ptrSTC->isARegexExclusiveRows(true);

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 4", "Ma ligne 1", 0);
  ptrSTO->editElement("Ma colonne 5", "Ma ligne 2", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addColumnForComparing("Ma colonne 1");
  ptrSTC->addColumnForComparing("Ma colonne 2");
  ptrSTC->addColumnForComparing("Ma colonne 3");

  // 3 inclus.
  ptrSTC->editRegexColumns("^.*[3-9]+$");
  ptrSTC->isARegexExclusiveColumns(true);

  ASSERT_TRUE(ptrSTC->compareWithReference());
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
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 3", 0);
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 4", 0);
  ptrSTO->editElement("Ma colonne 3", "Ma ligne 3", 0);
  ptrSTO->editElement("Ma colonne 4", "Ma ligne 4", 0);
  ptrSTO->editElement("Ma colonne 5", "Ma ligne 3", 0);
  ptrSTO->editElement("Ma colonne 3", "Ma ligne 1", 0);
  ptrSTO->editElement("Ma colonne 4", "Ma ligne 2", 0);
  ptrSTO->editElement("Ma colonne 3", "Ma ligne 3", 0);
  ptrSTO->editElement("Ma colonne 4", "Ma ligne 4", 0);
  ptrSTO->editElement("Ma colonne 3", "Ma ligne 5", 0);

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addColumnForComparing("Ma colonne 5");
  ptrSTC->editRegexColumns("^.*[1-2]+$");

  ptrSTC->editRegexRows("^.*[3-4]+$");
  ptrSTC->isARegexExclusiveRows(true);

  ASSERT_TRUE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testEpsilonColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{3478974.40208692104, 299512107932106.125});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{299538753331624.312, 3445501.01118461927});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 3478974.40208692150); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 1", 299512107932106.062); //NOK
  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 299538753331624.250); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 3445501.01118461974); //OK

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addEpsilonRow("Ma ligne 1", 1.0e-15);
  ptrSTC->addEpsilonRow("Ma ligne 2", 1.0e-15);

  ASSERT_TRUE(ptrSTC->compareWithReference());

  ptrSTC->addEpsilonRow("Ma ligne 1", 8.0e-17);
  ptrSTC->addEpsilonRow("Ma ligne 2", 1.0e-15);

  ASSERT_FALSE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testEpsilonRow()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{3478974.40208692104, 299512107932106.125});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{299538753331624.312, 3445501.01118461927});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 3478974.40208692150); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 1", 299512107932106.062); //NOK
  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 299538753331624.250); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 3445501.01118461974); //OK

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addEpsilonColumn("Ma colonne 1", 1.0e-15);
  ptrSTC->addEpsilonColumn("Ma colonne 2", 1.0e-15);

  ASSERT_TRUE(ptrSTC->compareWithReference());

  ptrSTC->addEpsilonColumn("Ma colonne 1", 8.0e-17);
  ptrSTC->addEpsilonColumn("Ma colonne 2", 1.0e-15);

  ASSERT_FALSE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testEpsilonRowColumn()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{3478974.40208692104, 299512107932106.125});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{299538753331624.312, 3445501.01118461927});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 3478974.40208692150); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 1", 299512107932106.062); //NOK
  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 299538753331624.250); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 3445501.01118461974); //OK

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addEpsilonColumn("Ma colonne 1", 1.0e-15);
  ptrSTC->addEpsilonColumn("Ma colonne 2", 8.0e-17);

  ptrSTC->addEpsilonRow("Ma ligne 1", 1.0e-15);
  ptrSTC->addEpsilonRow("Ma ligne 2", 1.0e-15);

  ASSERT_TRUE(ptrSTC->compareWithReference());

  ptrSTC->addEpsilonRow("Ma ligne 1", 8.0e-17);

  ASSERT_FALSE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testCompareOneElem()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{3478974.40208692104, 299512107932106.125});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{299538753331624.312, 3445501.01118461927});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeReferenceFile());

  ptrSTO->editElement("Ma colonne 1", "Ma ligne 1", 3478974.40208692150); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 1", 299512107932106.062); //NOK
  ptrSTO->editElement("Ma colonne 1", "Ma ligne 2", 299538753331624.250); //OK
  ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 3445501.01118461974); //OK

  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addEpsilonColumn("Ma colonne 1", 1.0e-15);
  ptrSTC->addEpsilonColumn("Ma colonne 2", 8.0e-17);

  ptrSTC->addEpsilonRow("Ma ligne 1", 8.0e-17);
  ptrSTC->addEpsilonRow("Ma ligne 2", 1.0e-15);

  ASSERT_TRUE(ptrSTC->compareElemWithReference("Ma colonne 1", "Ma ligne 1"));
  ASSERT_FALSE(ptrSTC->compareElemWithReference("Ma colonne 2", "Ma ligne 1"));
  ASSERT_TRUE(ptrSTC->compareElemWithReference("Ma colonne 1", "Ma ligne 2"));
  ASSERT_TRUE(ptrSTC->compareElemWithReference("Ma colonne 2", "Ma ligne 2"));

  ASSERT_FALSE(ptrSTC->compareWithReference());
}

void SimpleTableComparatorUnitTest::
testCompareWithElem()
{
  // Init STO
  ptrSTO->init("test", "dir_test");

  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{3478974.40208692104, 299512107932106.125});
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{299538753331624.312, 3445501.01118461927});


  // Init STC
  ptrSTC->init(ptrSTO);
  ASSERT_TRUE(ptrSTC->writeReferenceFile());
  ASSERT_TRUE(ptrSTC->readReferenceFile());

  ptrSTC->addEpsilonColumn("Ma colonne 1", 1.0e-15);
  ptrSTC->addEpsilonColumn("Ma colonne 2", 8.0e-17);

  ptrSTC->addEpsilonRow("Ma ligne 1", 8.0e-17);
  ptrSTC->addEpsilonRow("Ma ligne 2", 1.0e-15);

  ASSERT_TRUE(ptrSTC->compareElemWithReference(3478974.40208692150, "Ma colonne 1", "Ma ligne 1"));
  ASSERT_FALSE(ptrSTC->compareElemWithReference(299512107932106.062, "Ma colonne 2", "Ma ligne 1"));
  ASSERT_TRUE(ptrSTC->compareElemWithReference(299538753331624.250, "Ma colonne 1", "Ma ligne 2"));
  ASSERT_TRUE(ptrSTC->compareElemWithReference(3445501.01118461974, "Ma colonne 2", "Ma ligne 2"));
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

ARCANE_REGISTER_SERVICE_SIMPLETABLECOMPARATORUNITTEST(SimpleTableComparatorUnitTest, SimpleTableComparatorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
