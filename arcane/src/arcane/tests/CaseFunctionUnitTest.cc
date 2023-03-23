// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunctionUnitTest.cc                                     (C) 2000-2023 */
/*                                                                           */
/* Tests unitaires de 'ICaseFunction'.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ICaseFunction.h"
#include "arcane/core/ICaseMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de 'ICaseFunction'.
 */
class CaseFunctionUnitTest
: public BasicUnitTest
{
 public:

  explicit CaseFunctionUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  void _testLinearReal();
  void _checkValue(ICaseFunction* f, Real x, Real expected_y);
  void _checkValueEpsilon(ICaseFunction* f, Real x, Real expected_y);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(CaseFunctionUnitTest, IUnitTest, CaseFunctionUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseFunctionUnitTest::
CaseFunctionUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionUnitTest::
executeTest()
{
  _testLinearReal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionUnitTest::
_checkValue(ICaseFunction* f, Real x, Real expected_y)
{
  Real y = 0.0;
  f->value(x, y);
  if (y != expected_y)
    ARCANE_FATAL("Bad value func={0} x={1} y={2} expected={3}",
                 f->name(), x, y, expected_y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionUnitTest::
_checkValueEpsilon(ICaseFunction* f, Real x, Real expected_y)
{
  Real y = 0.0;
  f->value(x, y);
  if (!math::isNearlyEqual(y, expected_y))
    ARCANE_FATAL("Bad value func={0} x={1} y={2} expected={3}",
                 f->name(), x, y, expected_y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionUnitTest::
_testLinearReal()
{
  info() << "Test Linear Real";
  ICaseMng* cm = subDomain()->caseMng();
  String func_name = "test-real-linear";
  ICaseFunction* func = cm->findFunction(func_name);
  if (!func)
    ARCANE_FATAL("CaseFunction '{0}' not found", func_name);

  // Teste les valeurs exactes de la table
  _checkValue(func, -2.0, 2.0);
  _checkValue(func, 0.0, 2.0);
  _checkValue(func, 4.0, 7.0);
  _checkValue(func, 5.0, 31.0);
  _checkValue(func, 6.0, 50.0);
  _checkValue(func, 10.0, -1.0);
  _checkValue(func, 14.0, -3.0);
  _checkValue(func, 15.0, -3.0);

  // Test valeurs interpolées
  _checkValue(func, 2.0, 4.5);
  _checkValue(func, 4.5, 19.0);
  _checkValue(func, 7.0, 37.25);
  _checkValue(func, 8.0, 24.5);
  _checkValueEpsilon(func, 8.2, 21.95);
  _checkValueEpsilon(func, 8.0 + 1.0 / 3.0, 20.25);
  _checkValue(func, 9.0, 11.75);
  _checkValue(func, 12.0, -2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
