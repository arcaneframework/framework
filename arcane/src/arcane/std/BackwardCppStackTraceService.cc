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
#include "arcane/utils/Convert.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/AbstractService.h"

#include "arcane_packages.h"

//TODO : Ajouter les autres packages.
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
: public AbstractService
, public IStackTraceService
{

 public:

  explicit BackwardCppStackTraceService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_verbose_level(2)
  , m_human_readable(true)
  {}

 public:

  void build() override
  {
    // 0 : CallStack classique (nom de fonction uniquement)
    // 1 : CallStack classique avec numéro de ligne et fichier pour les classes/fonctions hors du namespace Arcane
    // 2 : (default) CallStack classique avec numéro de ligne et fichier pour toutes les classes/fonctions
    // 3 : CallStack classique avec numéro de ligne, fichier pour toutes les classes/fonctions et snippet pour les classes/fonctions hors du namespace Arcane
    // 4 : CallStack classique avec numéro de ligne, fichier et snippet pour toutes les classes/fonctions
    if (const auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CALLSTACK_VERBOSE", true)) {
      if (v.value() < 0 || v.value() > 4) {
        return;
      }
      m_verbose_level = v.value();
    }

    // Permet d'ajouter les espaces entre les appels dans la pile d'appel et
    // d'afficher le numéro de ligne avant le chemin du fichier source.
    // Sinon, on affiche le chemin du fichier et le numéro de ligne de manière
    // lisible par les debuggers/IDE (path:line).
    // Défaut = true.
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
