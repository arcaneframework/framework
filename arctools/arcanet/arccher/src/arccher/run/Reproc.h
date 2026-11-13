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
#ifndef ARCCHER_RUN_REPROC_H
#define ARCCHER_RUN_REPROC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/common/Array.h"
#include "arccore/trace/TraceAccessor.h"

#include "arccher/core/Arct.h"
#include "arccher/run/Run.h"

#include <string>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Reproc
: public Run
, public Arcane::TraceAccessor
{
  explicit Reproc(Arcane::ITraceMng* tm);

 public:

  void runCmd(Arct& runp, const Arcane::String& prog, const Arcane::String& cmdline) override;

 public:

  void cmdLineToArray(const Arcane::String& cmdline, Arcane::UniqueArray<std::string>& cmdlinearray);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
