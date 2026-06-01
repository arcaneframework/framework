// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BackwardCppStackTraceService.cc                             (C) 2000-2026 */
/*                                                                           */
/* Function call tracing service using 'backward-cpp'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Convert.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/AbstractService.h"

#include "arcane_packages.h"
#include "arccore/base/internal/DependencyInjection.h"

//TODO: Add the other packages.
#define BACKWARD_HAS_DW 1

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
: public TraceAccessor
, public IStackTraceService
{

 public:

  explicit BackwardCppStackTraceService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  , m_verbose_level(2)
  , m_human_readable(true)
  {}

  explicit BackwardCppStackTraceService(ITraceMng* tm)
  : TraceAccessor(tm)
  , m_verbose_level(2)
  , m_human_readable(true)
  {
    BackwardCppStackTraceService::build();
  }

 public:

  void build() override
  {
    // 0 : Classic CallStack (function name only)
    // 1 : Classic CallStack with line number and file for classes/functions outside the Arcane namespace
    // 2 : (default) Classic CallStack with line number and file for all classes/functions
    // 3 : Classic CallStack with line number, file for all classes/functions and snippet for classes/functions outside the Arcane namespace
    // 4 : Classic CallStack with line number, file and snippet for all classes/functions
    if (const auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CALLSTACK_VERBOSE", true)) {
      if (v.value() < 0 || v.value() > 4) {
        return;
      }
      m_verbose_level = v.value();
    }

    // Allows adding spaces between calls in the call stack and
    // displaying the line number before the source file path.
    // Otherwise, the file path and line number are displayed in a way
    // readable by debuggers/IDEs (path:line).
    // Default = true.
    if (const auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CALLSTACK_HUMAN_READABLE", true)) {
      m_human_readable = (v.value() != 0);
    }
  }

 public:

  StackTrace stackTrace(int first_function) override;
  StackTrace stackTraceFunction(int function_index) override;

 private:

  Int32 m_verbose_level;
  bool m_human_readable;
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

    if (m_verbose_level > 0 && src_line > 0) {
      bool arcane_function = String(trace.object_function).startsWith("Arcane");
      if (m_verbose_level > 1 || !arcane_function) {
        auto lines = sf.get_snippet(trace.source.filename, src_line, 5);
        if (m_human_readable) {
          message += "\n                  Line: ";
          message += src_line;
          message += " -- File: ";
          message += trace.source.filename;
        }
        else {
          message += "\n                  ";
          message += trace.source.filename;
          message += ":";
          message += src_line;
        }
        if (m_verbose_level > 3 || (m_verbose_level > 2 && !arcane_function)) {
          for (const auto& [line_num, line] : lines) {
            message += (line_num == src_line ? "\n                  >>>  " : "\n                       ");
            message += line_num;
            message += ":  ";
            message += line;
          }
        }
      }
    }

    if (m_human_readable)
      message += "\n\n";
    else
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

  if (st.size() < 1) {
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
                        ServiceProperty("BackwardCppStackTraceService", ST_Application),
                        ARCANE_SERVICE_INTERFACE(IStackTraceService));

ARCANE_DI_REGISTER_PROVIDER(BackwardCppStackTraceService,
                            DependencyInjection::ProviderProperty("BackwardCppStackTraceService"),
                            ARCANE_DI_INTERFACES(IStackTraceService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
