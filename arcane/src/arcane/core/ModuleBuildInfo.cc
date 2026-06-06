// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleBuildInfo.cc                                          (C) 2000-2019 */
/*                                                                           */
/* Parameters for building a module.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/ModuleBuildInfo.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleBuildInfo::
ModuleBuildInfo(ISubDomain* sd, IMesh* mesh, const String& name)
: m_sub_domain(sd)
, m_mesh_handle(mesh->handle())
, m_name(name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleBuildInfo::
ModuleBuildInfo(ISubDomain* sd, const MeshHandle& mesh_handle, const String& name)
: m_sub_domain(sd)
, m_mesh_handle(mesh_handle)
, m_name(name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleBuildInfo::
ModuleBuildInfo(ISubDomain* sd, const String& name)
: m_sub_domain(sd)
, m_mesh_handle(sd->defaultMeshHandle())
, m_name(name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
