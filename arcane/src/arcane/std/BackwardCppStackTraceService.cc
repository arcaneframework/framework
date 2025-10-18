// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BackwardCppStackTraceService.cc                             (C) 2000-2025 */
/*                                                                           */
/* Service de trace des appels de fonctions utilisant 'backward-cpp'.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/StackTrace.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/AbstractService.h"

#include "arcane_packages.h"

#define UNW_LOCAL_ONLY

#if defined(ARCANE_HAS_PACKAGE_DW)
#define BACKWARD_HAS_DW 1
#endif

#if defined(ARCANE_HAS_PACKAGE_LIBUNWIND)
#define BACKWARD_HAS_LIBUNWIND 1
#endif

#include "arcane/std/internal/backwardcpp/backward.hpp"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BackwardCppStackTraceService
: public AbstractService
, public IStackTraceService
{

 public:

  explicit BackwardCppStackTraceService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_hide_arcane_file_info(false)
  , m_hide_snippet(true)
  , m_hide_file_info(false)
  {}

 public:

  void build() override
  {
    // On retire les infos complémentaires (fichier/ligne/snippet) pour les appels Arcane uniquement.
    if (!platform::getEnvironmentVariable("ARCANE_CALLSTACK_NO_FILE_INFO_FOR_ARCANE_CALL").null()) {
      m_hide_arcane_file_info = true;
    }

    // On ajoute les snippets pour tous les appels.
    if (!platform::getEnvironmentVariable("ARCANE_CALLSTACK_SNIPPET").null()) {
      m_hide_snippet = false;
    }

    // On retire les infos complémentaires (fichier/ligne/snippet) pour tous les appels.
    if (!platform::getEnvironmentVariable("ARCANE_CALLSTACK_NO_FILE_INFO").null()) {
      m_hide_snippet = true;
      m_hide_file_info = true;
    }
  }

 public:

  StackTrace stackTrace(int first_function) override;
  StackTrace stackTraceFunction(int function_index) override;

 private:

  bool m_hide_arcane_file_info;
  bool m_hide_snippet;
  bool m_hide_file_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTrace BackwardCppStackTraceService::
stackTrace(int first_function)
{
  StringBuilder message;
  FixedStackFrameArray stack_frames;
  constexpr size_t hexa_buf_size = 100;
  char hexa[hexa_buf_size + 1];

  backward::StackTrace st;
  st.skip_n_firsts(first_function);
  st.load_here(64);

  backward::TraceResolver tr;
  tr.load_stacktrace(st);

  backward::SnippetFactory sf;

  for (size_t i = 0; i < st.size(); ++i) {
    backward::ResolvedTrace trace = tr.resolve(st[i]);
    message += "  ";
    snprintf(hexa, hexa_buf_size, "%14p", trace.addr);
    message += hexa;
    message += "  ";
    message += trace.object_function;

    UInt32 src_line = trace.source.line;
    if (!(m_hide_file_info || src_line == 0 || (m_hide_arcane_file_info && String(trace.object_function).startsWith("Arcane")))) {
      auto lines = sf.get_snippet(trace.source.filename, src_line, 5);
      message += "\n                  Line: ";
      message += src_line;
      message += " -- File: ";
      message += trace.source.filename;
      if (!m_hide_snippet) {
        for (const auto& [line_num, line] : lines) {
          message += (line_num == src_line ? "\n                  >>>  " : "\n                       ");
          message += line_num;
          message += ":  ";
          message += line;
        }
      }
    }
    message += "\n";

    stack_frames.addFrame(StackFrame(reinterpret_cast<intptr_t>(trace.addr)));
  }

  return { stack_frames, message };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTrace BackwardCppStackTraceService::
stackTraceFunction(int function_index)
{
  backward::StackTrace st;
  st.skip_n_firsts(function_index);
  st.load_here(1);

  if (st.size() != 1) {
    return {};
  }
  backward::TraceResolver tr;
  tr.load_stacktrace(st);

  backward::ResolvedTrace trace = tr.resolve(st[0]);

  return { trace.object_function };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(BackwardCppStackTraceService,
                        ServiceProperty("BackwardCpp", ST_Application),
                        ARCANE_SERVICE_INTERFACE(IStackTraceService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
