// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCriteriaLoadBalanceMng.h                                (C) 2000-2024 */
/*                                                                           */
/* Mesh load balance criteria manager.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHCRITERIALOADBALANCEMNG_H
#define ARCANE_CORE_MESHCRITERIALOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ILoadBalanceMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ICriteriaLoadBalanceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing criteria to be added to adjust
 *        the load balance.
 */
class ARCANE_CORE_EXPORT MeshCriteriaLoadBalanceMng
: public ICriteriaLoadBalanceMng
{
 public:

  MeshCriteriaLoadBalanceMng(ISubDomain* sd, const MeshHandle& mesh_handle);

 public:

  void addCriterion(VariableCellInt32& count) override;
  void addCriterion(VariableCellReal& count) override;
  void addMass(VariableCellInt32& count, const String& entity) override;
  void addCommCost(VariableFaceInt32& count, const String& entity) override;

  void reset() override;

  void setMassAsCriterion(bool active) override;
  void setNbCellsAsCriterion(bool active) override;
  void setCellCommContrib(bool active) override;
  void setComputeComm(bool active) override;

  Integer nbCriteria() override;

 private:

  ILoadBalanceMngInternal* m_internal;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
