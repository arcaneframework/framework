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

#include <arccore/base/Ref.h>

#include <arccore/common/CommandLineArguments.h>
#include <arccore/common/ExceptionUtils.h>

#include <arccore/trace/ITraceMng.h>

#include "arccher/core/Exception.h"
#include "arccher/run/Reproc.h"
#include "arccher/utils/ArctReader.h"
#include "arccher/utils/CmdLine.h"

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
    ArctReader reader(tm);
    reader.readJSON(cla);

    Arct* arct = reader.arct();

    CmdLine cmdline(tm, arct);
    Reproc run(tm, arct);

    for (const auto& prog : arct->m_run) {
      tm->info() << "prog : " << prog;
      String cmd = cmdline.buildCmdLine(prog);
      run.runCmd(prog, cmd);
    }

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
