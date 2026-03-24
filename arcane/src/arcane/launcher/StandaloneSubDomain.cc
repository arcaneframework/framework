// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneSubDomain.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Implémentation autonome d'un sous-domaine.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/StandaloneSubDomain.h"

#include "arccore/base/internal/ProfilingInternal.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Exception.h"

#include "arcane/impl/ArcaneSimpleExecutor.h"

#include "arcane/core/ISubDomain.h"

#include "arcane/launcher/ArcaneLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class StandaloneSubDomain::Impl
{
 public:

  Impl() = default;
  ~Impl()
  {
    arcaneCallFunctionAndCatchException([&]{ dumpStats(); });
    StandaloneSubDomain::_notifyRemoveStandaloneSubDomain();
  }

  void init(const String& case_file_name)
  {
    int r = m_simple_exec.initialize();
    if (r != 0)
      ARCANE_FATAL("Error during initialization r={0}", r);
    m_sub_domain = m_simple_exec.createSubDomain(case_file_name);
    m_trace_mng = makeRef(m_sub_domain->traceMng());
  }

  void dumpStats()
  {
    if (!m_sub_domain)
      return;
    std::ostringstream ostr;
    ::Arcane::Impl::dumpProfilingStatistics(ostr);
    if (m_trace_mng.get())
      m_trace_mng->info() << ostr.str();
    else
      std::cout << ostr.str() << "\n";
  }

 public:

  ArcaneSimpleExecutor m_simple_exec;
  ISubDomain* m_sub_domain = nullptr;
  Ref<ITraceMng> m_trace_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneSubDomain::
StandaloneSubDomain()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandaloneSubDomain::
_checkIsInitialized()
{
  if (m_p.get())
    return;
  ARCANE_FATAL("Instance of 'StandaloneSubDomain' is not initialized.\n"
               "You have to call ArcaneLauncher::createStandaloneSubDomain()\n"
               "to get a valid instance");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* StandaloneSubDomain::
traceMng()
{
  _checkIsInitialized();
  return m_p->m_trace_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* StandaloneSubDomain::
subDomain()
{
  _checkIsInitialized();
  return m_p->m_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandaloneSubDomain::
_initUniqueInstance(const String& case_file_name)
{
  m_p = makeRef(new Impl());
  m_p->init(case_file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StandaloneSubDomain::
_isValid()
{
  return m_p.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandaloneSubDomain::
_notifyRemoveStandaloneSubDomain()
{
  ArcaneLauncher::_notifyRemoveStandaloneSubDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
