// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.h                                          (C) 2000-2022 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINFOLISTVIEW_H
#define ARCANE_ITEMINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemFamily;
}

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste pour obtenir des informations sur les entités.
 */
class ARCANE_CORE_EXPORT ItemInfoListView
{
  friend class mesh::ItemFamily;

 public:

  ItemInfoListView() = default;
  explicit ItemInfoListView(IItemFamily* family);

 public:

  Item operator[](ItemLocalId local_id) const { return Item(ItemBase(ItemBaseBuildInfo(local_id.localId(), m_item_shared_info))); }

 private:

  // Seule ItemFamily peut créer des instances via ce constructeur
  ItemInfoListView(IItemFamily* family, ItemSharedInfo* shared_info, ItemInternalArrayView items_internal)
  : m_family(family)
  , m_item_shared_info(shared_info)
  , m_item_internal_list(items_internal)
  {}

 private:

  IItemFamily* m_family = nullptr;
  ItemSharedInfo* m_item_shared_info = nullptr;
  ItemInternalArrayView m_item_internal_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
