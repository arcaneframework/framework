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

#include "arcane/impl/MeshCriteriaLoadBalanceMng.h"
#include "arcane/impl/internal/LoadBalanceMngInternal.h"

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
addMass(VariableCellInt32& count, const String& entity)
{
  m_internal->addMass(count, m_mesh_handle.mesh(), entity);
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
addCommCost(VariableFaceInt32& count, const String& entity)
{
  m_internal->addCommCost(count, m_mesh_handle.mesh(), entity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshCriteriaLoadBalanceMng::
nbCriteria()
{
  return m_internal->nbCriteria(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
notifyEndPartition()
{
  m_internal->notifyEndPartition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
setMassAsCriterion(bool active)
{
  m_internal->setMassAsCriterion(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
setNbCellsAsCriterion(bool active)
{
  m_internal->setNbCellsAsCriterion(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
setCellCommContrib(bool active)
{
  m_internal->setCellCommContrib(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshCriteriaLoadBalanceMng::
cellCommContrib() const
{
  return m_internal->cellCommContrib(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMng::
setComputeComm(bool active)
{
  m_internal->setComputeComm(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableFaceReal& MeshCriteriaLoadBalanceMng::
commCost() const
{
  return m_internal->commCost(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& MeshCriteriaLoadBalanceMng::
massWeight() const
{
  return m_internal->massWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& MeshCriteriaLoadBalanceMng::
massResWeight() const
{
  return m_internal->massResWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellArrayReal& MeshCriteriaLoadBalanceMng::
mCriteriaWeight() const
{
  return m_internal->mCriteriaWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
