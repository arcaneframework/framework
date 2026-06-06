// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostItemsVariableParallelOperation.h                       (C) 2000-2025 */
/*                                                                           */
/* Parallel operations on ghost entities.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
#define ARCANE_CORE_PARALLEL_GHOSTITEMSVARIABLEPARALLELOPERATION_H
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
 * \brief Parallel operations on ghost entities.
 */
class ARCANE_CORE_EXPORT GhostItemsVariableParallelOperation
: public VariableParallelOperationBase
{
 public:

  explicit GhostItemsVariableParallelOperation(IItemFamily* family);
  virtual ~GhostItemsVariableParallelOperation() {} //!< Releases resources.

 public:
 protected:

  virtual void _buildItemsToSend();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
