// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/String.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/trace/ITraceMng.h"
#include "arccore/trace/TraceAccessor.h"
#include "arccore/trace/StandaloneTraceMessage.h"

#include <memory>
#include <sstream>

using namespace Arccore;

TEST(TraceMng, FatalMessage)
{
  ReferenceCounter<ITraceMng> tm(arccoreCreateDefaultTraceMng());
  TraceAccessor tr(tm.get());
  // Vérfie que FatalErrorException a bien le bon message.
  String message = "TestFatalError in utils";
  String new_message;
  bool is_ok = false;
  try{
    tr.fatal() << message;
  }
  catch(const FatalErrorException& ex)
  {
    new_message = ex.message();
    is_ok = true;
  }
  ASSERT_TRUE(is_ok) << "Exception not caught";

  ASSERT_TRUE(new_message==message) <<
  String::format("Bad message(wanted='{0}' current='{1}'",message,new_message);
}

TEST(TraceMng, FatalMessage2)
{
  ReferenceCounter<ITraceMng> tm(arccoreCreateDefaultTraceMng());
  TraceAccessor tr(tm.get());
  String message = "TestFatalErrorMessage in utils";
  String new_message;
  bool is_ok = false;
  try{
    tr.fatalMessage(StandaloneTraceMessage{} << message);
  }
  catch(const FatalErrorException& ex)
  {
    new_message = ex.message();
    is_ok = true;
  }
  ASSERT_TRUE(is_ok) << "Exception not caught";

  ASSERT_TRUE(new_message==message) <<
  String::format("Bad message(wanted='{0}' current='{1}'",message,new_message);
}
