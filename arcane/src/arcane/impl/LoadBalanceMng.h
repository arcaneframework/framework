// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceMng.h                                            (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire pour le partitionnement et l'équilibrage de charge.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_LOADBALANCEMNG_H
#define ARCANE_IMPL_LOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ILoadBalanceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implantation standard d'une interface d'enregistrement des variables
 * pour l'équilibrage de charge.
 *
 */
class ARCANE_IMPL_EXPORT LoadBalanceMng
: public ILoadBalanceMng
{
 public:

  explicit LoadBalanceMng(ISubDomain* sd);
  LoadBalanceMng(ISubDomain* sd, bool mass_as_criterion);

 public:

  /*!
   * Méthodes utilisées par les modules clients pour définir les critères
   * de partitionnement.
   */
  void addMass(VariableCellInt32& count, const String& entity="") override;
  void addCriterion(VariableCellInt32& count) override;
  void addCriterion(VariableCellReal& count) override;
  void addCommCost(VariableFaceInt32& count, const String& entity="") override;

  void reset() override;

  /*!
   * Méthodes utilisées par le MeshPartitioner pour accéder à la description
   * du problème.
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

 private:

  void _init(bool use_mass_as_criterion, bool is_legacy_init);
  static bool _isLegacyInit();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
