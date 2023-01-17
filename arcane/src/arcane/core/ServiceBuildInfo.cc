// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuildInfo.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Informations d'un service.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ServiceBuildInfo.h"
#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/IApplication.h"
#include "arcane/IMesh.h"
#include "arcane/ICaseOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(IApplication* app)
: m_application(app)
, m_service_parent(app)
, m_creation_type(ST_Application)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ISession* session)
: m_session(session)
, m_service_parent(session)
, m_creation_type(ST_Session)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(IMesh* mesh)
: ServiceBuildInfoBase(mesh->subDomain(),mesh->handle())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ISubDomain* sd,IMesh* mesh)
: ServiceBuildInfoBase(sd,mesh->handle())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ISubDomain* sd,const MeshHandle& mesh_handle)
: m_sub_domain(sd)
, m_mesh_handle(mesh_handle)
, m_service_parent(m_sub_domain)
, m_creation_type(ST_SubDomain)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(const MeshHandle& mesh_handle)
: ServiceBuildInfoBase(mesh_handle.subDomain(),mesh_handle)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ISubDomain* sd)
: m_sub_domain(sd)
, m_mesh_handle(sd->defaultMeshHandle())
, m_service_parent(sd)
, m_creation_type(ST_SubDomain)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ISubDomain* sd,ICaseOptions* co)
: m_sub_domain(sd)
, m_mesh_handle(co->meshHandle())
, m_case_options(co)
, m_service_parent(m_sub_domain)
, m_creation_type(ST_CaseOption)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfoBase::
ServiceBuildInfoBase(ICaseOptions* co)
: ServiceBuildInfoBase(co->subDomain(),co)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* ServiceBuildInfoBase::
mesh() const
{
  return m_mesh_handle.mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceBuildInfo::
ServiceBuildInfo(IServiceInfo* service_info, const ServiceBuildInfoBase& sbib)
: ServiceBuildInfoBase(sbib)
, m_service_info(service_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

