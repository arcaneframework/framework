// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractModule.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Classe gérant un module.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"

#include "arcane/AbstractModule.h"
#include "arcane/ISubDomain.h"
#include "arcane/ModuleBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractModule::
AbstractModule(const ModuleBuildInfo& vb)
: TraceAccessor(vb.subDomain()->traceMng())
, m_session(vb.subDomain()->session())
, m_sub_domain(vb.subDomain())
, m_default_mesh_handle(vb.meshHandle())
, m_name(vb.m_name)
, m_used(false)
, m_disabled(false)
, m_version_info(0,0,0)
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

