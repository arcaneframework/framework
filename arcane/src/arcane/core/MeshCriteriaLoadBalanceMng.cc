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

/*!
 * \brief Constructeur
 * \param sd Le sous-domaine où se trouve l'instance de ILoadBalanceMng.
 * \param mesh_handle Le maillage sur lequel les critères seront définis.
 */
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

void MeshCriteriaLoadBalanceMng::
setComputeComm(bool active)
{
  m_internal->setComputeComm(m_mesh_handle.mesh(), active);
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
