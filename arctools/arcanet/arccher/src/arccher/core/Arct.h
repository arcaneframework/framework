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
#ifndef ARCCHER_CORE_ARCT_H
#define ARCCHER_CORE_ARCT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/common/Array.h"

#include <string>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct CmdOpt
{
  Arcane::String m_opt = "{}";
  Arcane::String m_default = "";
  bool m_optional = true;
  bool m_needed = true;
};

struct CodeExecutor
{
  Arcane::String m_cmdline;
  std::unordered_map<std::string, CmdOpt> m_cmdopt;
};

struct RunParam2
{
  std::unordered_map<std::string, Arcane::String> m_properties;
  std::unordered_map<std::string, std::string> m_env_var;
};

struct Arct
{
  Arcane::UniqueArray<Arcane::String> m_run;

  RunParam2 m_global_params;
  std::unordered_map<std::string, RunParam2> m_specific_params;

  std::unordered_map<std::string, CodeExecutor> m_codes;
  std::unordered_map<std::string, CodeExecutor> m_executors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
