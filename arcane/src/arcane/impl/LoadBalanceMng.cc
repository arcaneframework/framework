// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMng.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Manager for partitioning and load balancing.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/LoadBalanceMng.h"

#include "arcane/utils/ValueConvert.h"

#include "arcane/impl/internal/LoadBalanceMngInternal.h"

#include "arcane/core/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool LoadBalanceMng::
_isLegacyInit()
{
  // If the environment variable is defined, use the historical initialization
  // (before version 3.14 of October 2024). This initialization by default used the
  // amount of memory allocated by the variables for partitioning, which meant that
  // the partitioning was not repeatable between check/release mode
  // or depending on the loaded modules (because the number of variables is different).
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LEGACY_INIT_LOADBALANCEMNG", true))
    return (v.value() != 0);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoadBalanceMng::
LoadBalanceMng(ISubDomain* sd)
: m_mesh_handle(sd->defaultMeshHandle())
{
  bool is_legacy_init = _isLegacyInit();
  // With the historical initialization, the default value is to use memory
  // as the criterion.
  bool use_mass_as_criterion = is_legacy_init;
  _init(use_mass_as_criterion, is_legacy_init);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LoadBalanceMng::
LoadBalanceMng(ISubDomain* sd, bool use_mass_as_criterion)
: m_mesh_handle(sd->defaultMeshHandle())
{
  bool is_legacy_init = _isLegacyInit();
  _init(use_mass_as_criterion, is_legacy_init);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
_init(bool use_mass_as_criterion, bool is_legacy_init)
{
  m_internal = makeRef(new LoadBalanceMngInternal(use_mass_as_criterion, is_legacy_init));
  m_internal->reset(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
reset()
{
  m_internal->reset(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
initAccess(IMesh* mesh)
{
  m_internal->initAccess(mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
endAccess()
{
  m_internal->endAccess();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addMass(VariableCellInt32& count, const String& entity)
{
  m_internal->addMass(count, m_mesh_handle.mesh(), entity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCriterion(VariableCellInt32& count)
{
  m_internal->addCriterion(count, m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCriterion(VariableCellReal& count)
{
  m_internal->addCriterion(count, m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
addCommCost(VariableFaceInt32& count, const String& entity)
{
  m_internal->addCommCost(count, m_mesh_handle.mesh(), entity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer LoadBalanceMng::
nbCriteria()
{
  return m_internal->nbCriteria(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
notifyEndPartition()
{
  m_internal->notifyEndPartition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
setMassAsCriterion(bool active)
{
  m_internal->setMassAsCriterion(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
setNbCellsAsCriterion(bool active)
{
  m_internal->setNbCellsAsCriterion(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
setCellCommContrib(bool active)
{
  m_internal->setCellCommContrib(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool LoadBalanceMng::
cellCommContrib() const
{
  return m_internal->cellCommContrib(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceMng::
setComputeComm(bool active)
{
  m_internal->setComputeComm(m_mesh_handle.mesh(), active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableFaceReal& LoadBalanceMng::
commCost() const
{
  return m_internal->commCost(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& LoadBalanceMng::
massWeight() const
{
  return m_internal->massWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellReal& LoadBalanceMng::
massResWeight() const
{
  return m_internal->massResWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VariableCellArrayReal& LoadBalanceMng::
mCriteriaWeight() const
{
  return m_internal->mCriteriaWeight(m_mesh_handle.mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ILoadBalanceMngInternal* LoadBalanceMng::
_internalApi()
{
  return m_internal.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ILoadBalanceMng*
arcaneCreateLoadBalanceMng(ISubDomain* sd)
{
  ILoadBalanceMng* lbm = new LoadBalanceMng(sd);
  return lbm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
