// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostItemsVariableParallelOperation.h                       (C) 2000-2008 */
/*                                                                           */
/* Opérations parallèles sur les entités fantômes.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
#define ARCANE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/VariableParallelOperationBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE_PARALLEL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations parallèle sur les entités fantômes.
 */
class ARCANE_CORE_EXPORT GhostItemsVariableParallelOperation
: public VariableParallelOperationBase
{
 public:

  GhostItemsVariableParallelOperation(IItemFamily* family);
  virtual ~GhostItemsVariableParallelOperation() {} //!< Libère les ressources.

 public:

 public:

 protected:

  virtual void _buildItemsToSend();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE_PARALLEL
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

