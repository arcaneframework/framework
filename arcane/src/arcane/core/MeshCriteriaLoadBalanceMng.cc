// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCriteriaLoadBalanceMng.cc                               (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des critères d'équilibre de charge des maillages.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshCriteriaLoadBalanceMng.h"
#include "arcane/core/internal/ILoadBalanceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCriteriaLoadBalanceMng::
MeshCriteriaLoadBalanceMng(ISubDomain* sd, const MeshHandle& mesh_handle)
: m_internal(sd->loadBalanceMng()->_internalApi())
, m_mesh_handle(mesh_handle)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
reset()
{
  m_internal->reset(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
addCriterion(VariableCellInt32& count)
{
  m_internal->addCriterion(count, m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
addCriterion(VariableCellReal& count)
{
  m_internal->addCriterion(count, m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
addMass(VariableCellInt32& count, const String& entity)
{
  m_internal->addMass(count, m_mesh_handle.mesh(), entity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
addCommCost(VariableFaceInt32& count, const String& entity)
{
  m_internal->addCommCost(count, m_mesh_handle.mesh(), entity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
