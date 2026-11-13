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

#include "arccher/utils/CmdLine.h"

#include "arccore/base/StringBuilder.h"
#include "arccore/trace/ITraceMng.h"
#include <arccore/base/FatalErrorException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CmdLine::
CmdLine(ITraceMng* tm)
: TraceAccessor(tm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CmdLine::
getVariable(RunParam2& code_params, CodeExecutor& code_executor, const String& variable_name)
{
  CmdOpt& cmd_opt = code_executor.m_cmdopt[variable_name.localstr()];

  String variable;
  if (code_params.m_properties.contains(variable_name.localstr())) {
    variable = code_params.m_properties[variable_name.localstr()];
  }
  else {
    if (!cmd_opt.m_optional) {
      ARCCORE_FATAL("{0} not optional", variable_name);
    }
    if (!cmd_opt.m_needed) {
      info() << "Var name : " << variable_name << " -- Value : ";
      return "";
    }
    if (cmd_opt.m_default.empty()) {
      ARCCORE_FATAL("{0} need default", variable_name);
    }
    variable = cmd_opt.m_default;
  }

  info() << "Var name : " << variable_name << " -- Value : " << variable;

  Int8 type_var = 0;
  StringBuilder new_cmd;
  Span<const Byte> cmd_prog = cmd_opt.m_opt.bytes();

  bool backslash = false;
  Integer pos_begin = -1;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {

    if (!backslash) {
      if (cmd_prog[i] == '{') {
        if (pos_begin == -2) {
          ARCCORE_FATAL("Error already {}");
        }
        pos_begin = i;
      }

      else if (cmd_prog[i] == '}') {
        if (pos_begin == -1) {
          ARCCORE_FATAL("Error {");
        }
        if (pos_begin + 1 != i) {
          ARCCORE_FATAL("Error not {}");
        }
        if (type_var == 1) {
          ARCCORE_FATAL("Error multiple {} not supported");
        }
        new_cmd += variable;
        pos_begin = -2;
        type_var = 1;
      }

      else if (pos_begin != -1) {
        ARCCORE_FATAL("Error not {}");
      }

      else {
        new_cmd += static_cast<char>(cmd_prog[i]);
      }
    }

    // Fin
    if (cmd_prog[i] == '\\') {
      backslash = true;
    }
  }

  if (pos_begin >= 0) {
    ARCCORE_FATAL("Error } -- Variable name : {0}", variable_name);
  }

  if (type_var == 0 && variable != "1") {
    info() << "Var name : " << variable_name << " -- Final value : ";
    return "";
  }
  info() << "Var name : " << variable_name << " -- Final value : " << new_cmd;
  return new_cmd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CmdLine::
replaceVariables(RunParam2& code_params, CodeExecutor& code_executor, const String& cmdline)
{
  info() << "cmdline : " << cmdline;
  StringBuilder new_cmd;
  Span<const Byte> cmd_prog = cmdline.bytes();

  bool backslash = false;
  Integer pos_begin = -1;
  UniqueArray<Byte> var_build;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {

    if (!backslash) {
      if (cmd_prog[i] == '{') {
        pos_begin = i;
      }

      else if (cmd_prog[i] == '}') {
        if (pos_begin == -1) {
          ARCCORE_FATAL("Error {");
        }
        new_cmd += getVariable(code_params, code_executor, Span<const Byte>(var_build));
        var_build.clear();
        pos_begin = -1;
      }

      else if (pos_begin != -1) {
        var_build.add(cmd_prog[i]);
      }

      else {
        new_cmd += static_cast<char>(cmd_prog[i]);
      }
    }

    // Fin
    if (cmd_prog[i] == '\\') {
      backslash = true;
    }
  }

  if (pos_begin != -1) {
    ARCCORE_FATAL("Error }");
  }
  return new_cmd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CmdLine::
buildCmdLine(Arct& runp, const String& prog)
{
  std::string prog_str = prog.localstr();

  CodeExecutor& code = runp.m_codes[prog_str];
  RunParam2& code_params = runp.m_specific_params[prog_str];

  CodeExecutor& exec = runp.m_executors[code_params.m_properties["executor_name"].localstr()];

  info() << "executor_name : " << code_params.m_properties["executor_name"];
  String cmd_prog = code.m_cmdline;
  String cmd_exec = exec.m_cmdline;

  auto aaa = replaceVariables(code_params, code, cmd_prog);
  info() << "aaa : " << aaa;

  code_params.m_properties["_CODE"] = aaa;

  auto bbb = replaceVariables(code_params, exec, cmd_exec);
  info() << "bbb : " << bbb;

  return bbb;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
