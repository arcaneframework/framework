// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/String.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/trace/ITraceMng.h"
#include "arccore/trace/TraceAccessor.h"

#include <memory>

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
