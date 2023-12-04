// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplaceTest.cc                                (C) 2000-2023 */
/*                                                                           */
/* Service de test de StringVariableReplace.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/core/internal/StringVariableReplace.h"

#include "arcane/tests/StringVariableReplaceTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringVariableReplaceTest
: public ArcaneStringVariableReplaceTestObject
{
 public:

  StringVariableReplaceTest(const ServiceBuildInfo& sbi);
  ~StringVariableReplaceTest();

 public:

  void initializeTest() override;
  void executeTest() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_STRINGVARIABLEREPLACETEST(StringVariableReplaceTest, StringVariableReplaceTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringVariableReplaceTest::
StringVariableReplaceTest(const ServiceBuildInfo& sbi)
: ArcaneStringVariableReplaceTestObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringVariableReplaceTest::
~StringVariableReplaceTest()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringVariableReplaceTest::
initializeTest()
{

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringVariableReplaceTest::
executeTest()
{
  ParameterList params;
  String test("TEST1");

  String result = StringVariableReplace::replaceWithCmdLineArgs(params, test);

  if(result != test) {
    ARCANE_FATAL("Test1 -- Expected: {0} -- Actual: {1}", test, result);
  }

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("ARCANE_REPLACE_SYMBOLS_IN_DATASET=1");

  result = StringVariableReplace::replaceWithCmdLineArgs(params, "");

  if(result != "") {
    ARCANE_FATAL("Test2 -- Expected: {0} -- Actual: {1}", "", result);
  }

  /*---------------------------------------------------------------------------*/

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);

  if(result != test) {
    ARCANE_FATAL("Test3 -- Expected: {0} -- Actual: {1}", test, result);
  }

  /*---------------------------------------------------------------------------*/

  test = "@TEST1@TEST2@@TEST3@TEST4";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);

  if(result != test) {
    ARCANE_FATAL("Test4 -- Expected: {0} -- Actual: {1}", test, result);
  }

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("TEST1=TEST5");

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  String expected = "TEST5TEST2@@TEST3@TEST4";

  if(result != expected) {
    ARCANE_FATAL("Test5 -- Expected: {0} -- Actual: {1}", expected, result);
  }

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("TEST2=TEST6");

  test = "@TEST1@TEST2@@TEST3@TEST4";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "TEST5TEST2@@TEST3@TEST4";

  if(result != expected) {
    ARCANE_FATAL("Test6 -- Expected: {0} -- Actual: {1}", expected, result);
  }

  /*---------------------------------------------------------------------------*/

  test = "@TEST1@@TEST2@@TEST3@TEST4";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "TEST5TEST6@TEST3@TEST4";

  if(result != expected) {
    ARCANE_FATAL("Test7 -- Expected: {0} -- Actual: {1}", expected, result);
  }

  /*---------------------------------------------------------------------------*/

  test = "@TEST1@@@@TEST2@@TEST3@TEST4@";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "TEST5@@TEST6@TEST3@TEST4@";

  if(result != expected) {
    ARCANE_FATAL("Test8 -- Expected: {0} -- Actual: {1}", expected, result);
  }

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("TEST4=TEST7");

  test = "@TEST1@@@@TEST2@@TEST3@TEST4@";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "TEST5@@TEST6@TEST3@TEST4@";

  if(result != expected) {
    ARCANE_FATAL("Test9 -- Expected: {0} -- Actual: {1}", expected, result);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
