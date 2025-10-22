// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StackTraceUnitTest.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Test du service de stack trace utilisé dans Application.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ServiceBuilder.h"

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
 * \brief Service de test des ItemVector
 */
class StackTraceUnitTest
: public BasicUnitTest
{
 public:

  explicit StackTraceUnitTest(const ServiceBuildInfo& cb);
  ~StackTraceUnitTest() override;

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  void _testStackTrace();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(StackTraceUnitTest,IUnitTest,StackTraceUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTraceUnitTest::
StackTraceUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTraceUnitTest::
~StackTraceUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StackTraceUnitTest::
executeTest()
{
  _testStackTrace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StackTraceUnitTest::
_testStackTrace()
{
  String s = platform::getStackTrace();
  info() << "CurrentStackTrace=" << s;

  StackTrace s2;
  if (platform::getStackTraceService()) {
    s2 = platform::getStackTraceService()->stackTraceFunction(0);
  }
  info() << "Last call=" << s2.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
