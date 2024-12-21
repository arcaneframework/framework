// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ILoadBalanceMngInternal.h                                   (C) 2000-2024 */
/*                                                                           */
/* Interface de classe interne gérant l'équilibre de charge des maillages.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_ILOADBALANCEMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_ILOADBALANCEMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief API interne à %Arcane de ILoadBalanceMng.
 *
 * Elle permet de conserver des critères d'équilibrage en fonction du maillage
 * ce qui est nécessaire pour les cas avec plusieurs maillages.
 */
class ARCANE_CORE_EXPORT ILoadBalanceMngInternal
{
 public:

  virtual ~ILoadBalanceMngInternal() = default; //!< Libère les ressources

 public:

  virtual void addMass(VariableCellInt32& count, IMesh* mesh, const String& entity) =0;
  virtual void addCriterion(VariableCellInt32& count, IMesh* mesh) =0;
  virtual void addCriterion(VariableCellReal& count, IMesh* mesh) =0;
  virtual void addCommCost(VariableFaceInt32& count, IMesh* mesh, const String& entity) =0;

 public:

  virtual void setMassAsCriterion(IMesh* mesh, bool active) =0;
  virtual void setNbCellsAsCriterion(IMesh* mesh, bool active) =0;
  virtual void setCellCommContrib(IMesh* mesh, bool active) = 0;
  virtual void setComputeComm(IMesh* mesh, bool active) = 0;
  virtual const VariableFaceReal& commCost(IMesh* mesh) = 0;
  virtual const VariableCellReal& massWeight(IMesh* mesh) = 0;
  virtual const VariableCellReal& massResWeight(IMesh* mesh) = 0;
  virtual const VariableCellArrayReal& mCriteriaWeight(IMesh* mesh) = 0;

  virtual bool cellCommContrib(IMesh* mesh) = 0;
  virtual Integer nbCriteria(IMesh* mesh) =0;

  virtual void reset(IMesh* mesh) = 0;
  virtual void initAccess(IMesh* mesh) = 0;
  virtual void endAccess() =0;
  virtual void notifyEndPartition() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
