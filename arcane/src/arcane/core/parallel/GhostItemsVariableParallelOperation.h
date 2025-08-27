// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostItemsVariableParallelOperation.h                       (C) 2000-2023 */
/*                                                                           */
/* Opérations parallèles sur les entités fantômes.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
#define ARCANE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/parallel/VariableParallelOperationBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations parallèle sur les entités fantômes.
 */
class ARCANE_CORE_EXPORT GhostItemsVariableParallelOperation
: public VariableParallelOperationBase
{
 public:

  explicit GhostItemsVariableParallelOperation(IItemFamily* family);
  virtual ~GhostItemsVariableParallelOperation() {} //!< Libère les ressources.

 public:

 protected:

  virtual void _buildItemsToSend();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

