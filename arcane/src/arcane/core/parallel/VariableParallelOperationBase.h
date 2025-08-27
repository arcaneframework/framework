// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableParallelOperationBase.h                             (C) 2000-2025 */
/*                                                                           */
/* Classe de base des opérations parallèle sur des variables.                */
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
 * \brief Interface d'une classe d'opérations parallèle sur des variables.
 *
 Ces opérations sont collectives.
 */
class ARCANE_CORE_EXPORT VariableParallelOperationBase
: public TraceAccessor
, public IVariableParallelOperation
{
 public:

  VariableParallelOperationBase(IParallelMng* pm);
  virtual ~VariableParallelOperationBase() {} //!< Libère les ressources.

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
  //! Liste des entités à envoyer à chaque processeur
  UniqueArray<SharedArray<ItemLocalId>> m_items_to_send;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

