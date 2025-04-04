// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodesOfItemReorderer.h                                      (C) 2000-2025 */
/*                                                                           */
/* Classe utilitaire pour réordonner les noeuds d'une entité.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_NODESOFITEMREORDERER_H
#define ARCANE_CORE_NODESOFITEMREORDERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SmallArray.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemTypeMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour réordonner les noeuds d'une entité.
 */
class ARCANE_CORE_EXPORT NodesOfItemReorderer
{
 public:

  NodesOfItemReorderer() = default;
  explicit NodesOfItemReorderer(ItemTypeMng* itm)
  : m_item_type_mng(itm)
  {}

 public:

  void setItemTypeMng(ItemTypeMng* itm) { m_item_type_mng = itm; }

 public:

  bool reorder(ItemTypeId type_id, ConstArrayView<Int64> nodes_uids);
  bool reorder1D(Int32 face_index, Int64 node_uid)
  {
    m_work_sorted_nodes.resize(1);
    m_work_sorted_nodes[0] = node_uid;
    return (face_index == 1);
  }
  ConstArrayView<Int64> sortedNodes() const { return m_work_sorted_nodes; }

 private:

  ItemTypeMng* m_item_type_mng = nullptr;
  SmallArray<Int64, 16> m_work_sorted_nodes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
