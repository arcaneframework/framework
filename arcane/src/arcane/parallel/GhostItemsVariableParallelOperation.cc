// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostItemsVariableParallelOperation.cc                      (C) 2000-2009 */
/*                                                                           */
/* Opérations parallèles sur les entités fantômes.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/parallel/GhostItemsVariableParallelOperation.h"

#include "arcane/ItemGroup.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_BEGIN_NAMESPACE_PARALLEL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostItemsVariableParallelOperation::
GhostItemsVariableParallelOperation(IItemFamily* family)
: VariableParallelOperationBase(family->parallelMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit la liste des entités à envoyer.
 *
 Il s'agit des entités fantômes.
*/
void GhostItemsVariableParallelOperation::
_buildItemsToSend()
{
  UniqueArray< SharedArray<ItemInternal*> >& items_to_send = _itemsToSend();

  ItemGroup all_items = itemFamily()->allItems();
  ENUMERATE_ITEM(iitem,all_items){
    const Item& item = *iitem;
    if (!item.isOwn()){
      items_to_send[item.owner()].add(item.internal());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE_PARALLEL
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
