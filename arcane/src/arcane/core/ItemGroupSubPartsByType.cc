// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupSubPartsByType.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Management of group sub-parts based on the type of its elements.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/ItemGroupInternal.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/MeshKind.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupSubPartsByType::
ItemGroupSubPartsByType(ItemGroupInternal* igi)
: m_group_internal(igi)
{
  // Checks if version 2 is used for ApplyOperationByBasicType
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_APPLYOPERATION_VERSION", true))
    m_use_v2_for_apply_operation = (v.value() == 2);

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DEBUG_APPLYOPERATION", true))
    m_is_debug_apply_operation = (v.value() > 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupSubPartsByType::
_initChildrenByTypeV2()
{
  bool is_verbose = m_is_debug_apply_operation;
  if (is_verbose)
    m_group_internal->mesh()->traceMng()->info() << "ItemGroupImpl::_initChildrenByTypeV2() name=" << m_group_internal->name();

  Int32 nb_basic_item_type = ItemTypeMng::nbBasicItemType();
  m_children_by_type_ids.resize(nb_basic_item_type);
  for (Integer i = 0; i < nb_basic_item_type; ++i) {
    m_children_by_type_ids[i] = UniqueArray<Int32>{ MemoryUtils::getDefaultDataAllocator() };
  }
}

void ItemGroupSubPartsByType::
_computeChildrenByTypeV2()
{
  ItemGroup that_group(m_group_impl);
  Int32 nb_item = m_group_internal->nbItem();
  IMesh* mesh = m_group_internal->mesh();
  ItemTypeMng* type_mng = mesh->itemTypeMng();
  ITraceMng* trace = mesh->traceMng();
  bool is_verbose = m_is_debug_apply_operation;
  if (is_verbose)
    trace->info() << "ItemGroupImpl::_computeChildrenByTypeV2 for " << m_group_internal->name();

  // If the mesh is Cartesian, we know that all entities have the same type
  if (nb_item > 0 && mesh->meshKind().meshStructure() == eMeshStructure::Cartesian) {
    ItemInfoListView lv(m_group_internal->m_item_family->itemInfoListView());
    m_unique_children_type = ItemTypeId{ lv.typeId(m_group_internal->itemsLocalId()[0]) };
    return;
  }

  Int32 nb_basic_item_type = ItemTypeMng::nbBasicItemType();
  m_unique_children_type = ItemTypeId{ IT_NullType };

  UniqueArray<Int32> nb_items_by_type(nb_basic_item_type);
  nb_items_by_type.fill(0);
  ENUMERATE_ (Item, iitem, that_group) {
    Item item = *iitem;
    Int16 item_type = item.type();
    if (item_type < nb_basic_item_type)
      ++nb_items_by_type[item_type];
  }
  const String& name = m_group_internal->name();
  Int32 nb_different_type = 0;
  for (Int32 i = 0; i < nb_basic_item_type; ++i) {
    m_children_by_type_ids[i].clear();
    const Int32 n = nb_items_by_type[i];
    m_children_by_type_ids[i].reserve(n);
    if (n > 0)
      ++nb_different_type;
    if (is_verbose)
      trace->info() << "ItemGroupImpl::_computeChildrenByTypeV2 for " << name
                    << " type=" << type_mng->typeName(i) << " nb=" << n;
  }
  if (is_verbose)
    trace->info() << "ItemGroupImpl::_computeChildrenByTypeV2 for " << name
                  << " nb_item=" << nb_item << " nb_different_type=" << nb_different_type;

  // If nb_different_type == 1, this means there is only one
  // entity type and we keep just that type, because in this case we will pass
  // the group directly as an argument to applyOperation().
  if (nb_item > 0 && nb_different_type == 1) {
    ItemInfoListView lv(m_group_internal->m_item_family->itemInfoListView());
    m_unique_children_type = ItemTypeId{ lv.typeId(m_group_internal->itemsLocalId()[0]) };
    if (is_verbose)
      trace->info() << "ItemGroupImpl::_computeChildrenByTypeV2 for " << name
                    << " unique_type=" << type_mng->typeName(m_unique_children_type);
    return;
  }

  ENUMERATE_ (Item, iitem, that_group) {
    Item item = *iitem;
    Integer item_type = item.type();
    if (item_type < nb_basic_item_type)
      m_children_by_type_ids[item_type].add(iitem.itemLocalId());
  }

  for (Int32 i = 0; i < nb_basic_item_type; ++i)
    applySimdPadding(m_children_by_type_ids[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
