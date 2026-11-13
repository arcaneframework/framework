// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.h                                                     (C) 2000-2026 */
/*                                                                           */
/* TODO.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccher/run/Reproc.h"

#include "arccher/core/Exception.h"
#include "arccore/trace/ITraceMng.h"
#include <arccore/base/FatalErrorException.h>

#include <reproc++/reproc.hpp>
#include <reproc++/run.hpp>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Reproc::
Reproc(ITraceMng* tm)
: TraceAccessor(tm)
{
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Reproc::
cmdLineToArray(const String& cmdline, UniqueArray<std::string>& cmdlinearray)
{
  Span<const Byte> cmd_prog = cmdline.bytes();

  Integer pos_begin = 0;
  bool previous_space = true;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {
    if (cmd_prog[i] == ' ' && previous_space) {
      pos_begin++;
    }
    else if (cmd_prog[i] == ' ') {
      String aaa = cmd_prog.subSpan(pos_begin, i - pos_begin);
      cmdlinearray.add(aaa.localstr());
      pos_begin = i + 1;
      previous_space = true;
    }
    else {
      previous_space = false;
    }
  }
  if (pos_begin != cmd_prog.size()) {
    String aaa = cmd_prog.subSpan(pos_begin, cmd_prog.size() - pos_begin);
    cmdlinearray.add(aaa.localstr());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Reproc::
runCmd(Arct& runp, const String& prog, const String& cmdline)
{
  std::string prog_local = prog.localstr();

  reproc::process process;
  reproc::options options;

  options.redirect.out.type = reproc::redirect::parent;
  options.redirect.err.type = reproc::redirect::parent;

  options.env.behavior = reproc::env::type::extend;

  if (!runp.m_specific_params[prog_local].m_env_var.empty())
    options.env.extra = runp.m_specific_params[prog_local].m_env_var;

  if (!runp.m_specific_params[prog_local].m_properties["working_dir"].empty())
    options.working_directory = runp.m_specific_params[prog_local].m_properties["working_dir"].localstr();

  UniqueArray<std::string> cmd;
  cmdLineToArray(cmdline, cmd);

  info() << "cmd : " << cmd;

  {
    std::error_code ec = process.start(cmd, options);
    if (ec) {
      throw std::system_error(ec);
    }
  }

  // auto [status, ec] = reproc::run(cmd, options);

  auto [status, ec] = process.wait(reproc::infinite);
  if (ec) {
    throw std::system_error(ec);
  }
  if (status) {
    throw RunException(status, prog_local);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
