// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariables.cc                                            (C) 2000-2005 */
/*                                                                           */
/* Connectivités communes aux maillages 1D, 2D et 3D                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/InvalidArgumentException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ISubDomain.h"
#include "arcane/ArcaneException.h"

#include "arcane/mesh/MeshVariables.h"
#include "arcane/Connectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshVariables::
MeshVariables(ISubDomain* sub_domain,const String& base_name)
: m_sub_domain(sub_domain)
, m_mesh_dimension(VariableBuildInfo(sub_domain,base_name+"MeshDimension"))
, m_mesh_connectivity(VariableBuildInfo(sub_domain,base_name+"MeshConnectivity"))
, m_item_families_name(VariableBuildInfo(sub_domain,base_name+"ItemFamiliesName"))
, m_item_families_kind(VariableBuildInfo(sub_domain,base_name+"ItemFamiliesKind"))
, m_parent_mesh_name(VariableBuildInfo(sub_domain,base_name+"ParentMeshName"))
, m_parent_group_name(VariableBuildInfo(sub_domain,base_name+"ParentGroupName"))
, m_child_meshes_name(VariableBuildInfo(sub_domain,base_name+"ChildMeshesName"))
{
  m_mesh_dimension = -1;
  m_mesh_connectivity = Connectivity::CT_Default;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

