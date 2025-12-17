// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputUnitTest.cc                                (C) 2000-2025 */
/*                                                                           */
/* Service de test pour les services implémentant ISimpleTableOutput.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/SimpleTableOutputUnitTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void SimpleTableOutputUnitTest::
_assertEqualArray(const UniqueArray<T>& expected, const UniqueArray<T>& actual)
{
  ASSERT_EQUAL(expected.size(), actual.size());
  for (Integer i = 0; i < actual.size(); i++) {
    ASSERT_EQUAL(expected[i], actual[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableOutputUnitTest::
setUpForClass()
{
  ptrSTO = options()->getSimpleTableOutput();
}

void SimpleTableOutputUnitTest::
setUp()
{
  ptrSTO->clear();
  ptrSTO->init();
}

void SimpleTableOutputUnitTest::
testInit()
{
  // La position de la première ligne et de la première colonne doivent être 0.
  ptrSTO->addRow("Ma ligne");
  ptrSTO->addColumn("Ma colonne");

  ASSERT_EQUAL(1, ptrSTO->numberOfRows());
  ASSERT_EQUAL(1, ptrSTO->numberOfColumns());

  ASSERT_EQUAL(0, ptrSTO->rowPosition("Ma ligne"));
  ASSERT_EQUAL(0, ptrSTO->columnPosition("Ma colonne"));

  ASSERT_TRUE(ptrSTO->editElement(0, 0, 123.));
  ASSERT_EQUAL(123., ptrSTO->element("Ma colonne", "Ma ligne"));

  ptrSTO->addColumn("Ma colonne 2");
  ASSERT_EQUAL(1, ptrSTO->columnPosition("Ma colonne 2"));
  ASSERT_TRUE(ptrSTO->editElement(1, 0, 456.));
  ASSERT_EQUAL(456., ptrSTO->element("Ma colonne 2", "Ma ligne"));
  ASSERT_EQUAL(123., ptrSTO->element("Ma colonne", "Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddRow1()
{
  ptrSTO->addRow("Ma ligne");
  ASSERT_EQUAL(1, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->numberOfColumns());
  ptrSTO->addRow("Ma seconde ligne");
  ASSERT_EQUAL(2, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->numberOfColumns());
}
void SimpleTableOutputUnitTest::
testAddRow2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  Integer position = ptrSTO->addRow("Ma ligne");
  ptrSTO->addColumn("Ma colonne 3");
  ASSERT_EQUAL(1, ptrSTO->numberOfRows());
  ASSERT_EQUAL(3, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->rowSize(position));
  ASSERT_EQUAL(0, ptrSTO->rowSize("Ma ligne"));
}
void SimpleTableOutputUnitTest::
testAddRow3()
{
  RealUniqueArray test = { 1, 2, 3, 4 };
  RealUniqueArray result = { 1, 2, 3 };
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addRow("Ma ligne", test);
  ptrSTO->addColumn("Ma colonne 4");
  _assertEqualArray(result, ptrSTO->row("Ma ligne"));
  ASSERT_EQUAL(3, ptrSTO->rowSize("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddRows1()
{
  ptrSTO->addRows(StringUniqueArray{ "L1", "L2", "L3", "L4" });
  ASSERT_EQUAL(4, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->numberOfColumns());
  ptrSTO->addRows(StringUniqueArray{ "L5" });
  ASSERT_EQUAL(5, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->numberOfColumns());
  ptrSTO->addRows(StringUniqueArray{});
  ASSERT_EQUAL(5, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->numberOfColumns());
}

void SimpleTableOutputUnitTest::
testAddColumn1()
{
  ptrSTO->addColumn("Ma colonne");
  ASSERT_EQUAL(1, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->numberOfRows());
  ptrSTO->addColumn("Ma seconde colonne");
  ASSERT_EQUAL(2, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->numberOfRows());
}
void SimpleTableOutputUnitTest::
testAddColumn2()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  Integer position = ptrSTO->addColumn("Ma colonne");
  ptrSTO->addRow("Ma ligne 3");
  ASSERT_EQUAL(1, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(3, ptrSTO->numberOfRows());
  ASSERT_EQUAL(0, ptrSTO->columnSize(position));
  ASSERT_EQUAL(0, ptrSTO->columnSize("Ma colonne"));
}
void SimpleTableOutputUnitTest::
testAddColumn3()
{
  RealUniqueArray test = { 1, 2, 3, 4 };
  RealUniqueArray result = { 1, 2, 3 };
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addColumn("Ma colonne", test);
  ptrSTO->addRow("Ma ligne 4");
  _assertEqualArray(result, ptrSTO->column("Ma colonne"));
  ASSERT_EQUAL(3, ptrSTO->columnSize("Ma colonne"));
}
void SimpleTableOutputUnitTest::
testAddColumn4()
{
  // On regarde si les valeurs des lignes non
  // utilisées sont à 0.
  RealUniqueArray result = { 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 8.0 };

  ptrSTO->addColumn("Ma colonne 1");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addElementInRow("Ma ligne 2", 2);
  ptrSTO->addElementInRow("Ma ligne 3", 3);
  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addElementInRow("Ma ligne 5", 5);
  ptrSTO->addElementInRow("Ma ligne 6", 6);
  ptrSTO->addElementInRow("Ma ligne 7", 7);
  ptrSTO->addElementInRow("Ma ligne 8", 8);

  ptrSTO->addElementInRow("Ma ligne 1", 0);
  ptrSTO->addElementInRow("Ma ligne 4", 0);

  for(Integer i = 2; i < 11; ++i){
    ptrSTO->addColumn("Ma colonne " + String::fromNumber(i));

    ptrSTO->addElementInRow("Ma ligne 2", 2);
    ptrSTO->addElementInRow("Ma ligne 5", 5);

    ptrSTO->addElementInRow("Ma ligne 3", 3);
    ptrSTO->addElementInRow("Ma ligne 6", 6);
    ptrSTO->addElementInRow("Ma ligne 7", 7);
    ptrSTO->addElementInRow("Ma ligne 8", 8);
  }

  for(Integer i = 1; i < 11; ++i){
    _assertEqualArray(result, ptrSTO->column("Ma colonne " + String::fromNumber(i)));
  }
}

void SimpleTableOutputUnitTest::
testAddColumns1()
{
  ptrSTO->addColumns(StringUniqueArray{ "C1", "C2", "C3", "C4" });
  ASSERT_EQUAL(4, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->numberOfRows());
  ptrSTO->addColumns(StringUniqueArray{ "C5" });
  ASSERT_EQUAL(5, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->numberOfRows());
  ptrSTO->addColumns(StringUniqueArray{});
  ASSERT_EQUAL(5, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(0, ptrSTO->numberOfRows());
}

void SimpleTableOutputUnitTest::
testAddElemRow1()
{
  Integer position = ptrSTO->addRow("Ma ligne");
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  RealUniqueArray result = { 1, 2, 3 };

  ASSERT_TRUE(ptrSTO->addElementInRow(position, 1));
  ASSERT_TRUE(ptrSTO->addElementInRow(position, 2));
  ASSERT_TRUE(ptrSTO->addElementInRow(position, 3));
  ASSERT_FALSE(ptrSTO->addElementInRow(position, 4));

  _assertEqualArray(result, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemRow2()
{
  ptrSTO->addRow("Ma ligne vide");
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  RealUniqueArray result = { 1, 2, 3 };

  ASSERT_FALSE(ptrSTO->addElementInRow("Ma ligne", 1, false));
  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne", 1));
  Integer position = ptrSTO->rowPosition("Ma ligne");
  ASSERT_TRUE(ptrSTO->addElementInRow(position, 2));
  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne", 3));
  ASSERT_FALSE(ptrSTO->addElementInRow("Ma ligne", 4));

  _assertEqualArray(result, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemSameRow1()
{
  ptrSTO->addRow("Ma ligne vide");
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  RealUniqueArray result = { 1, 2, 3, 4 };

  ASSERT_FALSE(ptrSTO->addElementInRow("Ma ligne", 1, false));
  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(2));

  Integer position = ptrSTO->rowPosition("Ma ligne");
  ASSERT_TRUE(ptrSTO->addElementInRow(position, 3));

  ptrSTO->addColumn("Ma colonne 4");
  ASSERT_TRUE(ptrSTO->addElementInSameRow(4));

  ASSERT_FALSE(ptrSTO->addElementInSameRow(5));

  _assertEqualArray(result, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemsRow1()
{
  Integer position = ptrSTO->addRow("Ma ligne");
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };
  ASSERT_TRUE(ptrSTO->addElementsInRow(position, test));
  _assertEqualArray(result1, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");

  ASSERT_FALSE(ptrSTO->addElementsInRow(position, test));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 6");

  ASSERT_FALSE(ptrSTO->addElementsInRow(position, test));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemsRow2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };

  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne", test));
  _assertEqualArray(result1, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");

  ASSERT_FALSE(ptrSTO->addElementsInRow("Ma ligne", test));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 6");

  Integer position = ptrSTO->rowPosition("Ma ligne");

  ASSERT_FALSE(ptrSTO->addElementsInRow(position, test));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemsSameRow1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };

  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne", test));
  _assertEqualArray(result1, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");

  ASSERT_FALSE(ptrSTO->addElementsInSameRow(test));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne"));

  ptrSTO->addColumn("Ma colonne 6");

  Integer position = ptrSTO->rowPosition("Ma ligne");

  ASSERT_FALSE(ptrSTO->addElementsInRow(position, test));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne"));
}

void SimpleTableOutputUnitTest::
testAddElemColumn1()
{
  Integer position = ptrSTO->addColumn("Ma colonne");
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result = { 1, 2, 3 };

  ASSERT_TRUE(ptrSTO->addElementInColumn(position, 1));
  ASSERT_TRUE(ptrSTO->addElementInColumn(position, 2));
  ASSERT_TRUE(ptrSTO->addElementInColumn(position, 3));
  ASSERT_FALSE(ptrSTO->addElementInColumn(position, 4));

  _assertEqualArray(result, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemColumn2()
{
  ptrSTO->addColumn("Ma colonne vide");
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result = { 1, 2, 3 };

  ASSERT_FALSE(ptrSTO->addElementInColumn("Ma colonne", 1, false));
  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne", 1));
  Integer position = ptrSTO->columnPosition("Ma colonne");
  ASSERT_TRUE(ptrSTO->addElementInColumn(position, 2));
  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne", 3));
  ASSERT_FALSE(ptrSTO->addElementInColumn("Ma colonne", 4));

  _assertEqualArray(result, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemSameColumn1()
{
  ptrSTO->addColumn("Ma colonne vide");
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result = { 1, 2, 3, 4 };

  ASSERT_FALSE(ptrSTO->addElementInColumn("Ma colonne", 1, false));
  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(2));

  Integer position = ptrSTO->columnPosition("Ma colonne");
  ASSERT_TRUE(ptrSTO->addElementInColumn(position, 3));

  ptrSTO->addRow("Ma ligne 4");
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(4));

  ASSERT_FALSE(ptrSTO->addElementInSameColumn(5));

  _assertEqualArray(result, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemsColumn1()
{
  Integer position = ptrSTO->addColumn("Ma colonne");
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };
  ASSERT_TRUE(ptrSTO->addElementsInColumn(position, test));
  _assertEqualArray(result1, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");

  ASSERT_FALSE(ptrSTO->addElementsInColumn(position, test));
  _assertEqualArray(result2, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 6");

  ASSERT_FALSE(ptrSTO->addElementsInColumn(position, test));
  _assertEqualArray(result3, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemsColumn2()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };

  ASSERT_TRUE(ptrSTO->addElementsInColumn("Ma colonne", test));
  _assertEqualArray(result1, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");

  ASSERT_FALSE(ptrSTO->addElementsInColumn("Ma colonne", test));
  _assertEqualArray(result2, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 6");

  Integer position = ptrSTO->columnPosition("Ma colonne");

  ASSERT_FALSE(ptrSTO->addElementsInColumn(position, test));
  _assertEqualArray(result3, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemsSameColumn1()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  RealUniqueArray test = { 1, 2, 3 };
  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 1, 2, 3, 1, 2 };
  RealUniqueArray result3 = { 1, 2, 3, 1, 2, 1 };

  ASSERT_TRUE(ptrSTO->addElementsInColumn("Ma colonne", test));
  _assertEqualArray(result1, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");

  ASSERT_FALSE(ptrSTO->addElementsInSameColumn(test));
  _assertEqualArray(result2, ptrSTO->column("Ma colonne"));

  ptrSTO->addRow("Ma ligne 6");

  Integer position = ptrSTO->columnPosition("Ma colonne");

  ASSERT_FALSE(ptrSTO->addElementsInColumn(position, test));
  _assertEqualArray(result3, ptrSTO->column("Ma colonne"));
}

void SimpleTableOutputUnitTest::
testAddElemSame1()
{
  ptrSTO->addRows(StringUniqueArray{ "Ma ligne 1", "Ma ligne 2", "Ma ligne 3", "Ma ligne 4", "Ma ligne 5" });
  ptrSTO->addColumns(StringUniqueArray{ "Ma colonne 1",
                                        "Ma colonne 2",
                                        "Ma colonne 3",
                                        "Ma colonne 4",
                                        "Ma colonne 5" });

  RealUniqueArray result1 = { 1 };
  RealUniqueArray result2 = { 2, 3 };
  RealUniqueArray result3 = { 0, 4, 5 };
  RealUniqueArray result4 = { 0, 0, 6, 7 };
  RealUniqueArray result5 = { 0, 0, 0, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(2));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(3));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(4));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(5));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(6));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(7));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(8));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(9));
  ASSERT_FALSE(ptrSTO->addElementInSameColumn(10));

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
  _assertEqualArray(result4, ptrSTO->row("Ma ligne 4"));
  _assertEqualArray(result5, ptrSTO->row("Ma ligne 5"));
}

void SimpleTableOutputUnitTest::
testEditElem1()
{
  ptrSTO->addColumn("Ma colonne 1");
  Integer posX1 = ptrSTO->addColumn("Ma colonne 2");
  Integer posX2 = ptrSTO->addColumn("Ma colonne 3");

  Integer posY0 = ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  Integer posY2 = ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  ASSERT_TRUE(ptrSTO->editElement(posX2, posY0, 10));
  ASSERT_TRUE(ptrSTO->editElement(posX1, posY2, 11));

  RealUniqueArray result1 = { 1, 2, 10 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 11, 9 };

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testEditElem2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  ASSERT_TRUE(ptrSTO->editElement("Ma colonne 3", "Ma ligne 1", 10));
  ASSERT_TRUE(ptrSTO->editElement("Ma colonne 2", "Ma ligne 3", 11));
  ASSERT_FALSE(ptrSTO->editElement("Ma colonne 4", "Ma ligne 1", 11));
  ASSERT_FALSE(ptrSTO->editElement("Ma colonne 1", "Ma ligne 4", 11));
  ASSERT_FALSE(ptrSTO->editElement("Ma colonne 4", "Ma ligne 4", 11));

  RealUniqueArray result1 = { 1, 2, 10 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 11, 9 };

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testElem1()
{
  Integer posX0 = ptrSTO->addColumn("Ma colonne 1");
  Integer posX1 = ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  Integer posY1 = ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  Integer posY2 = ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  ASSERT_EQUAL(5., ptrSTO->element(posX1, posY1));
  ASSERT_EQUAL(7., ptrSTO->element(posX0, posY2));
  ASSERT_EQUAL(8., ptrSTO->element(posX1, posY2));
}

void SimpleTableOutputUnitTest::
testElem2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  ASSERT_EQUAL(5., ptrSTO->element("Ma colonne 2", "Ma ligne 2"));
  ASSERT_EQUAL(7., ptrSTO->element("Ma colonne 1", "Ma ligne 3"));
  ASSERT_EQUAL(8., ptrSTO->element("Ma colonne 2", "Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testSizeRow1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  Integer posY0 = ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  Integer posY1 = ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4 });
  Integer posY2 = ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8 });

  ASSERT_EQUAL(3, ptrSTO->rowSize(posY0));
  ASSERT_EQUAL(1, ptrSTO->rowSize(posY1));
  ASSERT_EQUAL(2, ptrSTO->rowSize(posY2));

  ptrSTO->addColumn("Ma colonne 4", RealUniqueArray{ 9, 10, 11 });

  ASSERT_EQUAL(4, ptrSTO->rowSize(posY0));
  ASSERT_EQUAL(4, ptrSTO->rowSize(posY1));
  ASSERT_EQUAL(4, ptrSTO->rowSize(posY2));
}

void SimpleTableOutputUnitTest::
testSizeRow2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4 });
  ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8 });

  ASSERT_EQUAL(3, ptrSTO->rowSize("Ma ligne 1"));
  ASSERT_EQUAL(1, ptrSTO->rowSize("Ma ligne 2"));
  ASSERT_EQUAL(2, ptrSTO->rowSize("Ma ligne 3"));
  ASSERT_EQUAL(0, ptrSTO->rowSize("Ma ligne 4"));

  ptrSTO->addColumn("Ma colonne 4", RealUniqueArray{ 9, 10, 11 });

  ASSERT_EQUAL(4, ptrSTO->rowSize("Ma ligne 1"));
  ASSERT_EQUAL(4, ptrSTO->rowSize("Ma ligne 2"));
  ASSERT_EQUAL(4, ptrSTO->rowSize("Ma ligne 3"));
  ASSERT_EQUAL(0, ptrSTO->rowSize("Ma ligne 4"));
}

void SimpleTableOutputUnitTest::
testSizeColumn1()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  Integer posX0 = ptrSTO->addColumn("Ma colonne 1", RealUniqueArray{ 1, 2, 3 });
  Integer posX1 = ptrSTO->addColumn("Ma colonne 2", RealUniqueArray{ 4 });
  Integer posX2 = ptrSTO->addColumn("Ma colonne 3", RealUniqueArray{ 7, 8 });

  ASSERT_EQUAL(3, ptrSTO->columnSize(posX0));
  ASSERT_EQUAL(1, ptrSTO->columnSize(posX1));
  ASSERT_EQUAL(2, ptrSTO->columnSize(posX2));

  ptrSTO->addRow("Ma ligne 4", RealUniqueArray{ 9, 10, 11 });

  ASSERT_EQUAL(4, ptrSTO->columnSize(posX0));
  ASSERT_EQUAL(4, ptrSTO->columnSize(posX1));
  ASSERT_EQUAL(4, ptrSTO->columnSize(posX2));
}

void SimpleTableOutputUnitTest::
testSizeColumn2()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  ptrSTO->addColumn("Ma colonne 1", RealUniqueArray{ 1, 2, 3 });
  ptrSTO->addColumn("Ma colonne 2", RealUniqueArray{ 4 });
  ptrSTO->addColumn("Ma colonne 3", RealUniqueArray{ 7, 8 });

  ASSERT_EQUAL(3, ptrSTO->columnSize("Ma colonne 1"));
  ASSERT_EQUAL(1, ptrSTO->columnSize("Ma colonne 2"));
  ASSERT_EQUAL(2, ptrSTO->columnSize("Ma colonne 3"));
  ASSERT_EQUAL(0, ptrSTO->columnSize("Ma colonne 4"));

  ptrSTO->addRow("Ma ligne 4", RealUniqueArray{ 9, 10, 11 });

  ASSERT_EQUAL(4, ptrSTO->columnSize("Ma colonne 1"));
  ASSERT_EQUAL(4, ptrSTO->columnSize("Ma colonne 2"));
  ASSERT_EQUAL(4, ptrSTO->columnSize("Ma colonne 3"));
  ASSERT_EQUAL(0, ptrSTO->columnSize("Ma colonne 4"));
}

void SimpleTableOutputUnitTest::
testPosRowColumn1()
{
  Integer posX0 = ptrSTO->addColumn("Ma colonne 1");
  Integer posX1 = ptrSTO->addColumn("Ma colonne 2");
  Integer posX2 = ptrSTO->addColumn("Ma colonne 3");

  Integer posY0 = ptrSTO->addRow("Ma ligne 1", RealUniqueArray{ 1, 2, 3 });
  Integer posY1 = ptrSTO->addRow("Ma ligne 2", RealUniqueArray{ 4, 5, 6 });
  Integer posY2 = ptrSTO->addRow("Ma ligne 3", RealUniqueArray{ 7, 8, 9 });

  ASSERT_EQUAL(posX0, ptrSTO->columnPosition("Ma colonne 1"));
  ASSERT_EQUAL(posX1, ptrSTO->columnPosition("Ma colonne 2"));
  ASSERT_EQUAL(posX2, ptrSTO->columnPosition("Ma colonne 3"));

  ASSERT_EQUAL(posY0, ptrSTO->rowPosition("Ma ligne 1"));
  ASSERT_EQUAL(posY1, ptrSTO->rowPosition("Ma ligne 2"));
  ASSERT_EQUAL(posY2, ptrSTO->rowPosition("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testNumRowColumn1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");

  ASSERT_EQUAL(3, ptrSTO->numberOfColumns());
  ASSERT_EQUAL(2, ptrSTO->numberOfRows());
}

void SimpleTableOutputUnitTest::
testAddRowSameColumn1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(4));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(7));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 2));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(5));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(8));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 3));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(6));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(9));

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testAddRowSameColumn2()
{
  ptrSTO->addColumn("Ma colonne 1");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");
  ptrSTO->addRow("Ma ligne 6");
  ptrSTO->addRow("Ma ligne 7");
  ptrSTO->addRow("Ma ligne 8");
  ptrSTO->addRow("Ma ligne 9");

  RealUniqueArray result = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(2));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(3));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 4", 4));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(5));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(6));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(8));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(9));

  _assertEqualArray(result, ptrSTO->column("Ma colonne 1"));
}

void SimpleTableOutputUnitTest::
testAddRowSameColumn3()
{
  ptrSTO->addColumn("Ma colonne 1");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");
  ptrSTO->addRow("Ma ligne 6");
  ptrSTO->addRow("Ma ligne 7");
  ptrSTO->addRow("Ma ligne 8");
  ptrSTO->addRow("Ma ligne 9");

  RealUniqueArray result = { 1, 2, 3, 4, 0, 0, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(2));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(3));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(8));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(9));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 4", 4));
  ASSERT_FALSE(ptrSTO->addElementInSameColumn(5));
  ASSERT_FALSE(ptrSTO->addElementInSameColumn(6));

  _assertEqualArray(result, ptrSTO->column("Ma colonne 1"));
}

void SimpleTableOutputUnitTest::
testAddColumnSameRow1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(2));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(3));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 4));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(5));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(6));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(8));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(9));

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testAddColumnSameRow2()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");
  ptrSTO->addColumn("Ma colonne 6");
  ptrSTO->addColumn("Ma colonne 7");
  ptrSTO->addColumn("Ma colonne 8");
  ptrSTO->addColumn("Ma colonne 9");

  ptrSTO->addRow("Ma ligne 1");

  RealUniqueArray result = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(2));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(3));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 4", 4));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(5));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(6));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(8));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(9));

  _assertEqualArray(result, ptrSTO->row("Ma ligne 1"));
}

void SimpleTableOutputUnitTest::
testAddColumnSameRow3()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");
  ptrSTO->addColumn("Ma colonne 6");
  ptrSTO->addColumn("Ma colonne 7");
  ptrSTO->addColumn("Ma colonne 8");
  ptrSTO->addColumn("Ma colonne 9");

  ptrSTO->addRow("Ma ligne 1");

  RealUniqueArray result = { 1, 2, 3, 4, 0, 0, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(2));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(3));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(8));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(9));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 4", 4));
  ASSERT_FALSE(ptrSTO->addElementInSameRow(5));
  ASSERT_FALSE(ptrSTO->addElementInSameRow(6));

  _assertEqualArray(result, ptrSTO->row("Ma ligne 1"));
}

void SimpleTableOutputUnitTest::
testEditElemUDLR1()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");

  RealUniqueArray result1 = { 1, 2, 3 };
  RealUniqueArray result2 = { 4, 5, 6 };
  RealUniqueArray result3 = { 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->editElement("Ma colonne 2", "Ma ligne 2", 5));
  ASSERT_TRUE(ptrSTO->editElementDown(8));
  ASSERT_EQUAL(5., ptrSTO->elementUp());
  ASSERT_FALSE(ptrSTO->editElementDown(99));
  ASSERT_TRUE(ptrSTO->editElementLeft(7));
  ASSERT_EQUAL(8., ptrSTO->elementRight());
  ASSERT_FALSE(ptrSTO->editElementLeft(99));
  ASSERT_TRUE(ptrSTO->editElementUp(4));
  ASSERT_TRUE(ptrSTO->editElementUp(1));
  ASSERT_EQUAL(4., ptrSTO->elementDown());
  ASSERT_FALSE(ptrSTO->editElementUp(99));
  ASSERT_TRUE(ptrSTO->editElementRight(2));
  ASSERT_EQUAL(1., ptrSTO->elementLeft());
  ASSERT_TRUE(ptrSTO->editElementRight(3));
  ASSERT_FALSE(ptrSTO->editElementRight(99));
  ASSERT_TRUE(ptrSTO->editElementDown(6));
  ASSERT_TRUE(ptrSTO->editElementDown(9));
  ASSERT_FALSE(ptrSTO->editElementDown(99));
  ASSERT_EQUAL(6., ptrSTO->elementUp());

  _assertEqualArray(result1, ptrSTO->row("Ma ligne 1"));
  _assertEqualArray(result2, ptrSTO->row("Ma ligne 2"));
  _assertEqualArray(result3, ptrSTO->row("Ma ligne 3"));
}

void SimpleTableOutputUnitTest::
testEditElemDown1()
{
  // Voir testAddRowSameColumn3()
  ptrSTO->addColumn("Ma colonne 1");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addRow("Ma ligne 4");
  ptrSTO->addRow("Ma ligne 5");
  ptrSTO->addRow("Ma ligne 6");
  ptrSTO->addRow("Ma ligne 7");
  ptrSTO->addRow("Ma ligne 8");
  ptrSTO->addRow("Ma ligne 9");

  RealUniqueArray result = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(2));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(3));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(8));
  ASSERT_TRUE(ptrSTO->addElementInSameColumn(9));

  ASSERT_TRUE(ptrSTO->addElementInRow("Ma ligne 4", 4));
  ASSERT_TRUE(ptrSTO->editElementDown(5));
  ASSERT_TRUE(ptrSTO->editElementDown(6));

  _assertEqualArray(result, ptrSTO->column("Ma colonne 1"));
}

void SimpleTableOutputUnitTest::
testEditElemRight1()
{
  // Voir testAddColumnSameRow3()
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");
  ptrSTO->addColumn("Ma colonne 5");
  ptrSTO->addColumn("Ma colonne 6");
  ptrSTO->addColumn("Ma colonne 7");
  ptrSTO->addColumn("Ma colonne 8");
  ptrSTO->addColumn("Ma colonne 9");

  ptrSTO->addRow("Ma ligne 1");

  RealUniqueArray result = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 1", 1));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(2));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(3));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 7", 7));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(8));
  ASSERT_TRUE(ptrSTO->addElementInSameRow(9));

  ASSERT_TRUE(ptrSTO->addElementInColumn("Ma colonne 4", 4));
  ASSERT_TRUE(ptrSTO->editElementRight(5));
  ASSERT_TRUE(ptrSTO->editElementRight(6));

  _assertEqualArray(result, ptrSTO->row("Ma ligne 1"));
}

void SimpleTableOutputUnitTest::
testEditNameRow()
{
  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addRow("Ma ligne ratée");

  ptrSTO->addColumn("Ma colonne 1");

  RealUniqueArray test = { 1, 2, 3, 4 };

  ASSERT_TRUE(ptrSTO->addElementsInColumn("Ma colonne 1", test));

  ASSERT_TRUE(ptrSTO->editRowName("Ma ligne ratée", "Ma ligne 4"));

  ASSERT_EQUAL(ptrSTO->element("Ma colonne 1", "Ma ligne 4"), 4.);
}

void SimpleTableOutputUnitTest::
testEditNameColumn()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne ratée");

  ptrSTO->addRow("Ma ligne 1");

  RealUniqueArray test = { 1, 2, 3, 4 };

  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne 1", test));

  ASSERT_TRUE(ptrSTO->editColumnName("Ma colonne ratée", "Ma colonne 4"));

  ASSERT_EQUAL(ptrSTO->element("Ma colonne 4", "Ma ligne 1"), 4.);
}

void SimpleTableOutputUnitTest::
testWriteFile()
{
  ptrSTO->addColumn("Ma colonne 1");
  ptrSTO->addColumn("Ma colonne 2");
  ptrSTO->addColumn("Ma colonne 3");
  ptrSTO->addColumn("Ma colonne 4");

  ptrSTO->addRow("Ma ligne 1");
  ptrSTO->addRow("Ma ligne 2");
  ptrSTO->addRow("Ma ligne 3");
  ptrSTO->addRow("Ma ligne 4");

  ASSERT_TRUE(ptrSTO->addElementsInColumn("Ma colonne 1", RealUniqueArray{ 1, 5, 9, 13 }));
  ASSERT_TRUE(ptrSTO->addElementsInColumn("Ma colonne 2", RealUniqueArray{ 2, 6, 10, 14 }));
  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne 1", RealUniqueArray{ 3.1, 45678910.2345678 }));
  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne 2", RealUniqueArray{ 7.3, 8.4 }));
  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne 3", RealUniqueArray{ 11.5, 12.6 }));
  ASSERT_TRUE(ptrSTO->addElementsInRow("Ma ligne 4", RealUniqueArray{ 15.7, 16.8 }));

  ptrSTO->setPrecision(15);
  ptrSTO->setForcedToUseScientificNotation(false);
  ASSERT_TRUE(ptrSTO->writeFile("./testTable/", 0));
}

void SimpleTableOutputUnitTest::
tearDown()
{
  // N'est pas exécuté après un test qui a échoué.
}

void SimpleTableOutputUnitTest::
tearDownForClass()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLETABLEOUTPUTUNITTEST(SimpleTableOutputUnitTest, SimpleTableOutputUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
