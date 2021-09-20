// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractModule.cc                                           (C) 2000-2021 */
/*                                                                           */
/* Classe gérant un module.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/AbstractModule.h"
#include "arcane/ISubDomain.h"
#include "arcane/ModuleBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractModule::
AbstractModule(const ModuleBuildInfo& mbi)
: TraceAccessor(mbi.subDomain()->traceMng())
, m_session(mbi.subDomain()->session())
, m_sub_domain(mbi.subDomain())
, m_default_mesh_handle(mbi.meshHandle())
, m_name(mbi.m_name)
, m_used(false)
, m_disabled(false)
, m_version_info(0,0,0)
, m_accelerator_mng(mbi.subDomain()->acceleratorMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractModule::
~AbstractModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* AbstractModule::
parallelMng() const
{
  return m_sub_domain->parallelMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* AbstractModule::
traceMng() const
{
  return TraceAccessor::traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

