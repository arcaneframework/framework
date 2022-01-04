// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableParallelOperationBase.h                             (C) 2000-2009 */
/*                                                                           */
/* Classe de base des opérations parallèle sur des variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_VARIABLEPARALLELOPERATION_H
#define ARCANE_PARALLEL_VARIABLEPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/Parallel.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/VariableCollection.h"

#include "arcane/IVariableParallelOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class ItemInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE_PARALLEL

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

  virtual void build() {}

 public:

  virtual void setItemFamily(IItemFamily* family);
  virtual IItemFamily* itemFamily();
  virtual void addVariable(IVariable* variable);
  virtual void applyOperation(IDataOperation* operation);

 protected:

  UniqueArray< SharedArray<ItemInternal*> >& _itemsToSend()
  { return m_items_to_send; }

  virtual void _buildItemsToSend() =0;

 private:

  IParallelMng* m_parallel_mng;
  IItemFamily* m_item_family;
  VariableList m_variables;
  //! Liste des entités à envoyer à chaque processeur
  UniqueArray< SharedArray<ItemInternal*> > m_items_to_send;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE_PARALLEL
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

