// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostItemsVariableParallelOperation.cc                      (C) 2000-2023 */
/*                                                                           */
/* Parallel operations on ghost entities.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/GhostItemsVariableParallelOperation.h"

#include "arcane/ItemGroup.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

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
 * \brief Builds the list of entities to send.
 *
 * These are the ghost entities.
*/
void GhostItemsVariableParallelOperation::
_buildItemsToSend()
{
  auto& items_to_send = _itemsToSend();

  ItemGroup all_items = itemFamily()->allItems();
  ENUMERATE_ITEM(iitem,all_items){
    Item item = *iitem;
    if (!item.isOwn()){
      items_to_send[item.owner()].add(item);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
