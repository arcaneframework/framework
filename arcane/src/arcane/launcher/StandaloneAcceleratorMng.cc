// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneStandaloneAcceleratorMng.cc                       (C) 2000-2026 */
/*                                                                           */
/* Standalone implementation (without IApplication) of 'IAcceleratorMng.h'.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/StandaloneAcceleratorMng.h"

#include "arccore/concurrency/internal/ConcurrencyApplication.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/impl/MainFactory.h"
#include "arcane/impl/ArcaneMain.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/launcher/ArcaneLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  Impl::CoreArray<String>
  _stringListToCoreArray(const StringList& slist)
  {
    Impl::CoreArray<String> a;
    for (const String& s : slist)
      a.add(s);
    return a;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StandaloneAcceleratorMng::Impl
{
 public:

  Impl()
  {
    MainFactory main_factory;
    m_trace_mng = main_factory.createTraceMng();
    m_accelerator_mng = main_factory.createAcceleratorMngRef(m_trace_mng.get());

    const ApplicationInfo& app_info = ArcaneMain::defaultApplicationInfo();
    ApplicationBuildInfo& app_build_info = ArcaneMain::defaultApplicationBuildInfo();

    const CommandLineArguments& cmd_line_args = app_info.commandLineArguments();
    app_build_info.parseArgumentsAndSetDefaultsValues(cmd_line_args);
    app_build_info.setDefaultServices();

    {
      m_concurrency_application.setTraceMng(m_trace_mng);
      auto task_names = _stringListToCoreArray(app_build_info.taskImplementationServices());
      auto thread_names = _stringListToCoreArray(app_build_info.threadImplementationServices());
      ConcurrencyApplicationBuildInfo c(task_names.constView(), thread_names.constView(), app_build_info.nbTaskThread());
      m_concurrency_application.setCoreServices(c);
    }
  }

 public:

  ReferenceCounter<ITraceMng> m_trace_mng;
  Ref<IAcceleratorMng> m_accelerator_mng;
  ConcurrencyApplication m_concurrency_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneAcceleratorMng::
StandaloneAcceleratorMng()
: m_p(makeRef(new Impl()))
{
  m_p->m_accelerator_mng->initialize(ArcaneLauncher::acceleratorRuntimeInitialisationInfo());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* StandaloneAcceleratorMng::
traceMng() const
{
  return m_p->m_trace_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAcceleratorMng* StandaloneAcceleratorMng::
acceleratorMng() const
{
  return m_p->m_accelerator_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
