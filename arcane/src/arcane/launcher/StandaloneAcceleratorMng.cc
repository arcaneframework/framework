// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneArcaneLauncher.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Classe gérant une exécution simplifiée (sans sous-domaine).               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/StandaloneAcceleratorMng.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/impl/MainFactory.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/launcher/ArcaneLauncher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// TODO: à mettre dans IMainFactory
namespace Accelerator
{
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT Ref<IAcceleratorMng>
createAcceleratorMngRef(ITraceMng* tm);
}

class StandaloneAcceleratorMng::Impl
{
 public:
  Impl()
  {
    MainFactory main_factory;
    m_trace_mng = makeRef<ITraceMng>(main_factory.createTraceMng());
    m_accelerator_mng = Accelerator::createAcceleratorMngRef(m_trace_mng.get());
  }
 public:
  Ref<ITraceMng> m_trace_mng;
  Ref<IAcceleratorMng> m_accelerator_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneAcceleratorMng::
StandaloneAcceleratorMng()
: m_p(new Impl())
{
  {
    auto* runner = m_p->m_accelerator_mng->defaultRunner();
    arcaneInitializeRunner(*runner,m_p->m_trace_mng.get(),ArcaneLauncher::acceleratorRuntimeInitialisationInfo());
    m_p->m_accelerator_mng->initialize();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneAcceleratorMng::
~StandaloneAcceleratorMng()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* StandaloneAcceleratorMng::
traceMng()
{
  return m_p->m_trace_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAcceleratorMng* StandaloneAcceleratorMng::
acceleratorMng()
{
  return m_p->m_accelerator_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
