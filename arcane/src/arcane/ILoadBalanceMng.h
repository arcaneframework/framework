// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ILoadBalanceMng.h                                                (C) 2011 */
/*                                                                           */
/* Interface de description des caracteristiques du probleme pour le module  */
/* d'equilibrage de charge.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ILOADBALANCEMNG_H
#define ARCANE_ILOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'enregistrement des variables pour l'equilibrage de charge.
 *
 */
class ILoadBalanceMng
{
 public:

  virtual ~ILoadBalanceMng() {} //!< Libère les ressources.

 public:

  /*!
   * Methodes utilisees par les modules clients pour definir les criteres
   * de partitionnement.
   */
  virtual void addMass(VariableCellInt32& count, const String& entity="") =0;
  virtual void addCriterion(VariableCellInt32& count) =0;
  virtual void addCriterion(VariableCellReal& count) =0;
  virtual void addCommCost(VariableFaceInt32& count, const String& entity="") =0;

  virtual void reset() =0;

  /*!
   * Methodes utilisees par le MeshPartitioner pour acceder a la description
   * du probleme.
   */
  virtual void setMassAsCriterion(bool active=true) =0;
  virtual void setNbCellsAsCriterion(bool active=true) =0;
  virtual Integer nbCriteria() =0;
  virtual void setCellCommContrib(bool active=true) =0;
  virtual bool cellCommContrib() const =0;
  virtual void setComputeComm(bool active=true) =0;
  virtual void initAccess(IMesh *mesh=NULL) =0;
  const virtual VariableFaceReal& commCost() const =0;
  const virtual VariableCellReal& massWeight() const =0;
  const virtual VariableCellReal& massResWeight() const =0;
  const virtual VariableCellArrayReal& mCriteriaWeight() const =0;
  virtual void endAccess() =0;
  virtual void notifyEndPartition() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
