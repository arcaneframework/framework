// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyCompactPolicy.h                                   (C) 2000-2024 */
/*                                                                           */
/* Policy for compacting entities of a family.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILYCOMPACTPOLICY_H
#define ARCANE_MESH_ITEMFAMILYCOMPACTPOLICY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/IItemFamilyCompactPolicy.h"

#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Base class for entity compaction policies.
 *
 * This class is abstract because it does not implement IItemFamilyCompactPolicy::updateReferences().
 */
class ARCANE_MESH_EXPORT ItemFamilyCompactPolicy
: public TraceAccessor
, public IItemFamilyCompactPolicy
{
 public:

  explicit ItemFamilyCompactPolicy(ItemFamily* family);

 public:

  void beginCompact(ItemFamilyCompactInfos& compact_infos) override;
  void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos) override;
  void endCompact(ItemFamilyCompactInfos& compact_infos) override;
  void finalizeCompact(IMeshCompacter* compacter) override
  {
    ARCANE_UNUSED(compacter);
  }
  IItemFamily* family() const override;
  void compactConnectivityData() override;

 protected:

  void _changeItem(Int32* items, Integer nb_item, Int32ConstArrayView old_to_new)
  {
    for (Integer i = 0; i < nb_item; ++i) {
      Integer item_local_id = items[i];
      items[i] = old_to_new[item_local_id];
    }
  }

 protected:

  ItemFamily* _family() const { return m_family; }

 private:

  ItemFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compaction policy for Node, Edge, Face, or Cell entity families.
 */
class StandardItemFamilyCompactPolicy
: public ItemFamilyCompactPolicy
{
 public:

  StandardItemFamilyCompactPolicy(ItemFamily* family);

 public:

  void updateInternalReferences(IMeshCompacter* compacter) override;

 public:

  IItemFamily* m_node_family;
  IItemFamily* m_edge_family;
  IItemFamily* m_face_family;
  IItemFamily* m_cell_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
