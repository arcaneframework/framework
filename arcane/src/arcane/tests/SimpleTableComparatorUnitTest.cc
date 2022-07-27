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
