// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCriteriaLoadBalanceMng.h                                (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des critères d'équilibre de charge des maillages.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_MESHCRITERIALOADBALANCEMNG_H
#define ARCANE_IMPL_MESHCRITERIALOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ILoadBalanceMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ICriteriaLoadBalanceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implantation standard d'une interface d'enregistrement des variables
 * pour l'equilibrage de charge.
 *
 */
class ARCANE_IMPL_EXPORT MeshCriteriaLoadBalanceMng
: public ICriteriaLoadBalanceMng
{
 public:

  MeshCriteriaLoadBalanceMng(ISubDomain* sd, const MeshHandle& mesh_handle);

 public:

  void addMass(VariableCellInt32& count, const String& entity) override;
  void addCriterion(VariableCellInt32& count) override;
  void addCriterion(VariableCellReal& count) override;
  void addCommCost(VariableFaceInt32& count, const String& entity) override;

  void setMassAsCriterion(bool active) override;
  void setNbCellsAsCriterion(bool active) override;
  void setCellCommContrib(bool active) override;
  void setComputeComm(bool active) override;
  const VariableFaceReal& commCost() const override;
  const VariableCellReal& massWeight() const override;
  const VariableCellReal& massResWeight() const override;
  const VariableCellArrayReal& mCriteriaWeight() const override;

  bool cellCommContrib() const override;
  Integer nbCriteria() override;

  void reset() override;

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
