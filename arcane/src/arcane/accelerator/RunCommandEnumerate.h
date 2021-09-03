// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandEnumerate.h                                       (C) 2000-2021 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une liste d'entités.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunQueueInternal.h"

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType>
class ItemRunCommand
{
 public:
  ItemRunCommand(RunCommand& command,const ItemVectorViewT<ItemType>& items)
  : m_command(command), m_items(items)
  {
  }
  RunCommand& m_command;
  ItemVectorViewT<ItemType> m_items;
};

template<typename ItemType> ItemRunCommand<ItemType>
operator<<(RunCommand& command,const ItemGroupT<ItemType>& items)
{
  return ItemRunCommand<ItemType>(command,items.view());
}

template<typename ItemType> ItemRunCommand<ItemType>
operator<<(RunCommand& command,const ItemVectorViewT<ItemType>& items)
{
  return ItemRunCommand<ItemType>(command,items);
}

template<typename ItemType,typename Lambda>
void operator<<(ItemRunCommand<ItemType>&& nr,Lambda f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}
template<typename ItemType,typename Lambda>
void operator<<(ItemRunCommand<ItemType>& nr,Lambda f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer un groupe d'entités
#define RUNCOMMAND_ENUMERATE(ItemNameType,iter_name,item_group)         \
  A_FUNCINFO << item_group << [=] ARCCORE_HOST_DEVICE (ItemNameType##LocalId iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
