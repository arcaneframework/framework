// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMng.h                                            (C) 2000-2024 */
/*                                                                           */
/* Module standard de description du probleme pour l'equilibrage de charge.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_LOADBALANCEMNG_H
#define ARCANE_IMPL_LOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ILoadBalanceMng.h"

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
class ARCANE_IMPL_EXPORT LoadBalanceMng
: public ILoadBalanceMng
{
 public:

  explicit LoadBalanceMng(ISubDomain* sd, bool massAsCriterion = true);

 public:
  /*!
   * Methodes utilisees par les modules clients pour definir les criteres
   * de partitionnement.
   */
  void addMass(VariableCellInt32& count, const String& entity="") override;
  void addCriterion(VariableCellInt32& count) override;
  void addCriterion(VariableCellReal& count) override;
  void addCommCost(VariableFaceInt32& count, const String& entity="") override;

  void reset() override;

  /*!
   * Methodes utilisees par le MeshPartitioner pour acceder a la description
   * du probleme.
   */
  void setMassAsCriterion(bool active = true) override;
  void setNbCellsAsCriterion(bool active = true) override;
  void setCellCommContrib(bool active = true) override;
  bool cellCommContrib() const override;
  void setComputeComm(bool active = true) override;
  Integer nbCriteria() override;
  void initAccess(IMesh* mesh=nullptr) override;
  const VariableFaceReal& commCost() const override;
  const VariableCellReal& massWeight() const override;
  const VariableCellReal& massResWeight() const override;
  const VariableCellArrayReal& mCriteriaWeight() const override;
  void endAccess() override;
  void notifyEndPartition() override;

  ILoadBalanceMngInternal* _internalApi() override;

 private:

  Ref<ILoadBalanceMngInternal> m_internal;
  MeshHandle m_mesh_handle;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
