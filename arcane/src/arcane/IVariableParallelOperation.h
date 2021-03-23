// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableParallelOperation.h                                (C) 2000-2006 */
/*                                                                           */
/* Interface d'une classe d'opérations parallèles sur des variables.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IVARIABLEPARALLELOPERATION_H
#define ARCANE_IVARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamily;
class IVariable;
class IDataOperation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une classe d'opérations parallèle sur des variables.
 *
 Ces opérations sont collectives.
 
 Avant d'effectuer l'opération, il faut positionner la famille
 d'entités (setItemFamily()),
 puis ajouter la liste des variables sur lesquelles seront effectuées
 les opérations.
 */
class IVariableParallelOperation
{
 public:

  virtual ~IVariableParallelOperation() {} //!< Libère les ressources.

 public:

  virtual void build() =0; //!< Construit l'instance

 public:

  /*! \brief Positionne la famille d'entité sur laquelle on souhaite opérer.
   * 
   Le maillage doit être positionner avant d'ajouter des variables.
   Il ne peut l'être qu'une seule fois.
  */
  virtual void setItemFamily(IItemFamily* family) =0;
  
  //! Famille d'entités sur laquelle on opère
  virtual IItemFamily* itemFamily() =0;

  //! Ajoute \a variable à la liste des variables concernées par l'opération
  virtual void addVariable(IVariable* variable) =0;

  /*!
   * \brief Applique l'opération.
   */
  virtual void applyOperation(IDataOperation* operation) =0;
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

