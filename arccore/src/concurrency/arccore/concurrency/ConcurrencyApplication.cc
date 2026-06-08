// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyApplication.h                                    (C) 2000-2026 */
/*                                                                           */
/* Management of multi-threading services for an Arccore application.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/internal/ConcurrencyApplication.h"

#include "arccore/base/CheckedConvert.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/internal/DependencyInjection.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/concurrency/ITaskImplementation.h"
#include "arccore/concurrency/IThreadImplementation.h"
#include "arccore/concurrency/IThreadImplementationService.h"
#include "arccore/concurrency/TaskFactory.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*!
 * Tries to instantiate a service implementing \a InterfaceType with
 * the list of service names \a names. Returns the found instance
 * if it exists and fills \a found_name (if not null) with the instance name.
 * As soon as an instance is found, it is returned.
 * Returns \a nullptr if no instance is available.
 */
template <typename InterfaceType> Ref<InterfaceType> ConcurrencyApplication::
tryCreateServiceUsingInjector(ConstArrayView<String> names, String* found_name, bool has_trace)
{
  DependencyInjection::Injector injector;
  injector.fillWithGlobalFactories();
  // Adds an instance of ITraceMng* for services that need it
  if (has_trace)
    injector.bind(m_trace.get());

  if (found_name)
    (*found_name) = String();
  for (String s : names) {
    auto t = injector.createInstance<InterfaceType>(s, true);
    if (t.get()) {
      if (found_name)
        (*found_name) = s;
      return t;
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConcurrencyApplication::
setCoreServices(const ConcurrencyApplicationBuildInfo& build_info)
{
  // Searches for the service used to know the call stack
  bool has_dbghelp = false;
  {
    String dbghelp_service_name = "DbgHelpStackTraceService";
    Impl::CoreArray<String> names;
    String found_name;
    Ref<IStackTraceService> sv;
    const auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_BACKWARDCPP", true);
    if (v && v.value() != 0) {
      names.add("BackwardCppStackTraceService");
    }
    names.add("LibUnwind");
    names.add("DbgHelpStackTraceService");
    sv = tryCreateServiceUsingInjector<IStackTraceService>(names.constView(), &found_name, true);
    if (found_name == dbghelp_service_name)
      has_dbghelp = true;
    if (sv.get()) {
      m_stack_trace_service = sv;
      Platform::setStackTraceService(sv.get());
    }
  }

  // Searches for the service used to know the symbol information
  // of the source code. For now, only LLVM is supported and this service
  // is activated
  // if the environment variable ARCANE_LLVMSYMBOLIZER_PATH is defined.
  {
    Impl::CoreArray<String> names;
    String found_name;
    Ref<ISymbolizerService> sv;

    if (!platform::getEnvironmentVariable("ARCANE_LLVMSYMBOLIZER_PATH").null())
      names.add("LLVMSymbolizer");
    if (has_dbghelp)
      names.add("DbgHelpSymbolizerService");
    sv = tryCreateServiceUsingInjector<ISymbolizerService>(names.constView(), &found_name, true);
    if (sv.get()) {
      m_symbolizer_service = sv;
      Platform::setSymbolizerService(sv.get());
    }
  }

  // Searches for the service implementing multi-threading support.
  {
    ConstArrayView<String> names = build_info.threadImplementationServices();
    String found_name;
    auto sv = tryCreateServiceUsingInjector<IThreadImplementationService>(names, &found_name, false);
    if (!sv.get())
      ARCCORE_FATAL("Can not find implementation for 'IThreadImplementation' (names='{0}').", names);
    m_thread_implementation_service = sv;
    m_thread_implementation = sv->createImplementation();
    Arccore::Concurrency::setThreadImplementation(m_thread_implementation.get());
    m_thread_implementation->initialize();
    m_used_thread_service_name = found_name;
  }

  // The thread manager may have changed and therefore
  // the trace manager must be reinitialized.
  m_trace->resetThreadStatus();

  // Searches for the service used to manage tasks
  {
    Integer nb_task_thread = build_info.nbTaskThread();
    if (nb_task_thread >= 0) {

      ConstArrayView<String> names = build_info.taskImplementationServices();
      String found_name;
      auto sv = tryCreateServiceUsingInjector<ITaskImplementation>(names, &found_name, false);
      if (sv.get()) {
        TaskFactoryInternal::setImplementation(sv.get());
        sv->initialize(nb_task_thread);
        m_used_task_service_name = found_name;
        m_task_implementation = sv;
      }
      else
        ARCCORE_FATAL("Can not find task implementation service (names='{0}')."
                      " Please check if Arcane is configured with Intel TBB library",
                      names);
    }

    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_TASK_VERBOSE_LEVEL", true))
      TaskFactory::setVerboseLevel(v.value());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
