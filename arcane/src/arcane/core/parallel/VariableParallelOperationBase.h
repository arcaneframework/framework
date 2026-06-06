// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableParallelOperationBase.h                             (C) 2000-2025 */
/*                                                                           */
/* Base class for parallel operations on variables.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_VARIABLEPARALLELOPERATION_H
#define ARCANE_CORE_PARALLEL_VARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/core/IVariableParallelOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a class of parallel operations on variables.
 *
 These operations are collective.
 */
class ARCANE_CORE_EXPORT VariableParallelOperationBase
: public TraceAccessor
, public IVariableParallelOperation
{
 public:

  VariableParallelOperationBase(IParallelMng* pm);
  virtual ~VariableParallelOperationBase() {} //!< Frees resources.

 public:

  void build() override {}

 public:

  void setItemFamily(IItemFamily* family) override;
  IItemFamily* itemFamily() override;
  void addVariable(IVariable* variable) override;
  void applyOperation(IDataOperation* operation) override;

 protected:

  Array<SharedArray<ItemLocalId>>& _itemsToSend() { return m_items_to_send; }

  virtual void _buildItemsToSend() = 0;

 private:

  IParallelMng* m_parallel_mng;
  IItemFamily* m_item_family;
  VariableList m_variables;
  //! List of entities to send to each processor
  UniqueArray<SharedArray<ItemLocalId>> m_items_to_send;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
