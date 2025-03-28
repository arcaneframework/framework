// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplaceTest.cc                                (C) 2000-2025 */
/*                                                                           */
/* Service de test de StringVariableReplace.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParameterList.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/List.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/BasicUnitTest.h"
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

  explicit StringVariableReplaceTest(const ServiceBuildInfo& sbi);
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
  StringList args;
  platform::fillCommandLineArguments(args);
  const CommandLineArguments cla{ args };

  ParameterList params(cla.parameters());
  String test("aaaa");
  bool fatal = true;

  String result = StringVariableReplace::replaceWithCmdLineArgs(test);

  if (result != test) {
    ARCANE_FATAL("Test 1 -- Expected: {0} -- Actual: {1}", test, result);
  }
  info() << "Test 1 OK";

  /*---------------------------------------------------------------------------*/

  result = StringVariableReplace::replaceWithCmdLineArgs("");

  if (result != "") {
    ARCANE_FATAL("Test 2 -- Expected: {0} -- Actual: {1}", "", result);
  }
  info() << "Test 2 OK";

  /*---------------------------------------------------------------------------*/

  result = StringVariableReplace::replaceWithCmdLineArgs(test);

  if (result != test) {
    ARCANE_FATAL("Test 3 -- Expected: {0} -- Actual: {1}", test, result);
  }
  info() << "Test 3 OK";

  /*---------------------------------------------------------------------------*/

  //params.addParameterLine("aa=bb"); // Dans le CMakeLists.

  /*---------------------------------------------------------------------------*/

  test = "aa@aa";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 4 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(test, false, false);
  String expected = "aaaa";

  if (result != expected) {
    ARCANE_FATAL("Test 4 -- Expected: {0} -- Actual: {1}", test, result);
  }
  info() << "Test 4 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@aa@aa@";

  result = StringVariableReplace::replaceWithCmdLineArgs(test);
  expected = "bbaabb";

  if (result != expected) {
    ARCANE_FATAL("Test 5 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 5 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@@aa@";

  result = StringVariableReplace::replaceWithCmdLineArgs(test);
  expected = "bbbb";

  if (result != expected) {
    ARCANE_FATAL("Test 6 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 6 OK";

  /*---------------------------------------------------------------------------*/

  test = "@";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 7 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(test, false, false);
  expected = "";

  if (result != expected) {
    ARCANE_FATAL("Test 7 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 7 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "bb";

  if (result != expected) {
    ARCANE_FATAL("Test 8 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 8 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@aa@aa";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(params, test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 9 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test, false, false);
  expected = "bbaaaa";

  if (result != expected) {
    ARCANE_FATAL("Test 9 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 9 OK";

  /*---------------------------------------------------------------------------*/

  test = "@cc@aa";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(params, test, true);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 10 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test, false, false);
  expected = "aa";

  if (result != expected) {
    ARCANE_FATAL("Test 10 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 10 OK";

  /*---------------------------------------------------------------------------*/

  test = "aa\\@bb.fr";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "aa@bb.fr";

  if (result != expected) {
    ARCANE_FATAL("Test 11 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 11 OK";

  /*---------------------------------------------------------------------------*/

  test = "@@@";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(params, test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 12 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test, false, false);
  expected = "";

  if (result != expected) {
    ARCANE_FATAL("Test 12 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 12 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@aa@@aa@aa";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(params, test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 13 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test, false, false);
  expected = "bbaaaaaa";

  if (result != expected) {
    ARCANE_FATAL("Test 13 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 13 OK";

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("cc=dd");

  /*---------------------------------------------------------------------------*/

  test = "@aa@@cc@";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "bbdd";

  if (result != expected) {
    ARCANE_FATAL("Test 14 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 14 OK";

  /*---------------------------------------------------------------------------*/

  test = "@aa@@@cc@";

  fatal = false;
  try {
    StringVariableReplace::replaceWithCmdLineArgs(params, test);
    fatal = true;
  }
  catch (const FatalErrorException& e) {
  }
  if (fatal) {
    ARCANE_FATAL("Test 15 no fatal test");
  }

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test, false, false);
  expected = "bbcc";

  if (result != expected) {
    ARCANE_FATAL("Test 15 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 15 OK";

  /*---------------------------------------------------------------------------*/

  params.addParameterLine("ee=ff@gg");

  /*---------------------------------------------------------------------------*/

  test = "@aa@aa@ee@";

  result = StringVariableReplace::replaceWithCmdLineArgs(params, test);
  expected = "bbaaff@gg";

  if (result != expected) {
    ARCANE_FATAL("Test 16 -- Expected: {0} -- Actual: {1}", expected, result);
  }
  info() << "Test 16 OK";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
