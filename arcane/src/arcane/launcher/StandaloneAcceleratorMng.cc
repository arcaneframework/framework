// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneStandaloneAcceleratorMng.cc                       (C) 2000-2025 */
/*                                                                           */
/* Implémentation autonome (sans IApplication) de 'IAcceleratorMng.h'.       */
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

class StandaloneAcceleratorMng::Impl
{
 public:
  Impl()
  {
    MainFactory main_factory;
    m_trace_mng = makeRef<ITraceMng>(main_factory.createTraceMng());
    m_accelerator_mng = main_factory.createAcceleratorMngRef(m_trace_mng.get());
  }
 public:
  Ref<ITraceMng> m_trace_mng;
  Ref<IAcceleratorMng> m_accelerator_mng;
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
