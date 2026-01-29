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
#ifndef ARCCORE_COMMON_INTERNAL_CONCURRENCYAPPLICATION_H
#define ARCCORE_COMMON_INTERNAL_CONCURRENCYAPPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arccore/base/String.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/Ref.h"
#include "arccore/base/CoreArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_CONCURRENCY_EXPORT ConcurrencyApplicationBuildInfo
{
 public:

  ConcurrencyApplicationBuildInfo(ConstArrayView<String> task_service_names,
                                  ConstArrayView<String> thread_service_names,
                                  Int32 nb_task)
  : m_task_implementation_services(task_service_names)
  , m_thread_implementation_services(thread_service_names)
  , m_nb_task(nb_task)
  {
  }

 public:

  ConstArrayView<String> taskImplementationServices() const
  {
    return m_task_implementation_services.constView();
  }

  ConstArrayView<String> threadImplementationServices() const
  {
    return m_thread_implementation_services.constView();
  }

  Int32 nbTaskThread() const { return m_nb_task; }

 private:

  Impl::CoreArray<String> m_task_implementation_services;
  Impl::CoreArray<String> m_thread_implementation_services;
  Int32 m_nb_task = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_CONCURRENCY_EXPORT ConcurrencyApplication
{
 public:

  void setTraceMng(ReferenceCounter<ITraceMng> tm) { m_trace = tm; }
  void setCoreServices(const ConcurrencyApplicationBuildInfo& build_info);

  template <typename InterfaceType> Ref<InterfaceType>
  tryCreateServiceUsingInjector(ConstArrayView<String> names, String* found_name, bool has_trace);

 public:

  ReferenceCounter<ITraceMng> m_trace; //!< Gestionnaire de traces
  Ref<IStackTraceService> m_stack_trace_service;
  Ref<ISymbolizerService> m_symbolizer_service;
  Ref<IThreadImplementationService> m_thread_implementation_service;
  Ref<IThreadImplementation> m_thread_implementation;
  Ref<ITaskImplementation> m_task_implementation;
  //! Nom du service utilisé pour gérer les threads
  String m_used_thread_service_name;
  //! Nom du service utilisé pour gérer les tâches
  String m_used_task_service_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
