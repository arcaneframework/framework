// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.cc                                                     (C) 2000-2026 */
/*                                                                           */
/* TODO.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccher/core/Arct.h"

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/PlatformUtils.h>
#include <arccore/base/Ref.h>
#include <arccore/base/String.h>

#include <arccore/common/CommandLineArguments.h>
#include <arccore/common/JSONReader.h>
#include <arccore/common/ArcaNetReader.h>
#include <arccore/common/ExceptionUtils.h>

#include <arccore/trace/ITraceMng.h>

#include <reproc++/reproc.hpp>
#include <reproc++/run.hpp>

#include <fstream>
#include <unordered_map>

#include "arccher/core/Exception.h"

using namespace Arcane;
using namespace Arccher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  Ref<ITraceMng> tm_ref;
  ITraceMng* tm = nullptr;

  try {
    tm_ref = makeRef(arccoreCreateDefaultTraceMng());
    tm = static_cast<ITraceMng*>(ARCCORE_CHECK_POINTER(tm_ref.get()));
    tm->pushTraceClass("ArcCher");
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, nullptr, false);
  }
  catch (...) {
    return ExceptionUtils::print(nullptr, false);
  }

  try {
    CommandLineArguments cla(&argc, &argv);
    // Run runp;
    // readJSON(runp, cla, tm);
    // run(runp, tm);
    Arct arct;
  }
  catch (const RunException& e) {
    std::cout << std::endl;
    tm->info() << "Can't continue with the execution.";
    tm->info() << "Program: " << e.prog();
    tm->info() << "Process return: " << e.code();
    return e.code();
  }
  catch (const std::system_error& e) {
    std::cout << std::endl;
    tm->info() << "Can't continue with the execution.";
    tm->info() << "Process return: " << e.code().value();
    tm->info() << "Process message: " << e.code().message();
    tm->info() << "Process what: " << e.what();
    return e.code().value();
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, tm, false);
  }
  catch (...) {
    return ExceptionUtils::print(tm, false);
  }

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
