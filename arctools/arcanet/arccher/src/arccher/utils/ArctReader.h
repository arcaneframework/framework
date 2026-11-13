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
#ifndef ARCCHER_UTILS_ARCTREADER_H
#define ARCCHER_UTILS_ARCTREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/common/Array.h"
#include "arccore/trace/TraceAccessor.h"

#include "arccher/core/Arct.h"

#include <string>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArctReader
: public Arcane::TraceAccessor
{
 public:

  explicit ArctReader(Arcane::ITraceMng* tm);
  ~ArctReader() override = default;

 public:

  void readJSON(Arcane::CommandLineArguments& cla);
  void readFile(const Arcane::String& file, Arcane::JSONDocument& doc, Arcane::JSONValue& root);
  void readCommonPart(RunParam2& runp, const Arcane::JSONValue& arccher_part);
  void firstReadArccherGeneralPart(const Arcane::JSONValue& arccher_part);
  void firstReadArccherCodePart(const Arcane::JSONValue& arccher_part, const Arcane::String& code);
  void readCodesExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const Arcane::JSONValue& code_or_exec_part, Arcane::ArrayView<Arcane::String> needed_executors);
  void readExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const Arcane::JSONValue& arccher_part, Arcane::ArrayView<Arcane::String> needed_executors);
  void readCodes(std::unordered_map<std::string, CodeExecutor>& codes_executors, const Arcane::JSONValue& arccher_part, Arcane::ArrayView<Arcane::String> needed_executors);

  Arct* arct() const { return m_run.get(); }

 private:

  bool _readAllFile(Arcane::StringView filename, bool is_binary, Arcane::Array<Arcane::Byte>& out_bytes);
  bool _JSONValueAsBool(const Arcane::JSONValue& value);

 private:

  Arcane::Ref<Arct> m_run;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
