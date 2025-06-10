// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableParallelOperation.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface d'une classe d'opérations parallèles sur des variables.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEPARALLELOPERATION_H
#define ARCANE_CORE_IVARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une classe d'opérations parallèle sur des variables.
 *
 * Ces opérations sont collectives.
 *
 * Avant d'effectuer l'opération, il faut positionner la famille
 * d'entités (setItemFamily()),
 * puis ajouter la liste des variables sur lesquelles seront effectuées
 * les opérations.
 */
class ARCANE_CORE_EXPORT IVariableParallelOperation
{
 public:

  virtual ~IVariableParallelOperation() = default; //!< Libère les ressources.

 public:

  virtual void build() =0; //!< Construit l'instance

 public:

  /*!
   * \brief Positionne la famille d'entité sur laquelle on souhaite opérer.
   * 
   * La famille doit être positionnée avant d'ajouter des variables.
   * Elle ne peut l'être qu'une seule fois.
   */
  virtual void setItemFamily(IItemFamily* family) =0;
  
  //! Famille d'entités sur laquelle on opère
  virtual IItemFamily* itemFamily() =0;

  //! Ajoute \a variable à la liste des variables concernées par l'opération
  virtual void addVariable(IVariable* variable) =0;

  //! Applique l'opération.
  virtual void applyOperation(IDataOperation* operation) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

