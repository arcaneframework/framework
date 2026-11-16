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
#ifndef ARCCHER_UTILS_CMDLINE_H
#define ARCCHER_UTILS_CMDLINE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"
#include "arccore/trace/TraceAccessor.h"

#include "arccher/core/ArccherGlobal.h"
#include "arccher/core/Arct.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCHER_EXPORT CmdLine
: public Arcane::TraceAccessor
{
 public:

  explicit CmdLine(Arcane::ITraceMng* tm, Arct* runp);
  Arcane::String getVariable(RunParam2& code_params, CodeExecutor& code_executor, const Arcane::String& variable_name);
  Arcane::String replaceVariables(RunParam2& code_params, CodeExecutor& code_executor, const Arcane::String& cmdline);
  Arcane::String buildCmdLine(const Arcane::String& prog);

 public:

  ~CmdLine() override = default;

 private:

  Arct* m_runp;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
