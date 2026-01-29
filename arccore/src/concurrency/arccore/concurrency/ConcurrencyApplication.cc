// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyApplication.h                                    (C) 2000-2026 */
/*                                                                           */
/* Gestion des services de multi-threading d'une application Arccore.        */
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
 * Essaie d'instancier un service implémentant \a InterfaceType avec
 * la liste de nom de services \a names.  Retourne l'instance trouvée
 * si elle existe et remplit \a found_name (si non nul) avec le nom de
 * l'instance. Dès qu'une instance est trouvée, on la retourne.
 * Retourne \a nullptr si aucune instance n'est disponible.
 */
template <typename InterfaceType> Ref<InterfaceType> ConcurrencyApplication::
tryCreateServiceUsingInjector(ConstArrayView<String> names, String* found_name, bool has_trace)
{
  DependencyInjection::Injector injector;
  injector.fillWithGlobalFactories();
  // Ajoute une instance de ITraceMng* pour les services qui en ont besoin
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

  // Recherche le service utilisé pour connaitre la pile d'appel
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

  // Recherche le service utilisé pour connaitre les infos sur les symboles
  // du code source. Pour l'instant, on ne supporte que LLVM et on n'active ce service
  // que si la variable d'environnement ARCANE_LLVMSYMBOLIZER_PATH est définie.
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

  // Recherche le service implémentant le support du multi-threading.
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

  // Le gestionnaire de thread a pu changer et il faut donc
  // reinitialiser le gestionnaire de trace.
  m_trace->resetThreadStatus();

  // Recherche le service utilisé pour gérer les tâches
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
