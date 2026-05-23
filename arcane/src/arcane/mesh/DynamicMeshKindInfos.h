// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshKindInfos.h                                      (C) 2000-2025 */
/*                                                                           */
/* Mesh information for a given entity kind.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHKINDINFOS_H
#define ARCANE_MESH_DYNAMICMESHKINDINFOS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Event.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/VariableTypedef.h"

#include "arcane/mesh/ItemInternalMap.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Define to disable events if one wishes to test
// the influence on performance (there are supposedly none).
#define ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mesh information for a given entity kind.
 *
 * An instance of this class manages all mesh structures
 * for a given entity kind.
 */
class ARCANE_MESH_EXPORT DynamicMeshKindInfos
: public TraceAccessor
{
 public:

  // TODO: to be removed
  typedef Arcane::mesh::ItemInternalMap ItemInternalMap;

 private:

  using ItemInternalMapData = ItemInternalMap::BaseData;

 public:

  //! Creates an instance for a given mesh and kind.
  DynamicMeshKindInfos(IMesh* mesh,eItemKind kind,const String& kind_name);
  //! Frees resources
  ~DynamicMeshKindInfos();

 public:

  void build();
  
  //! Reallocates and recalculates structures after mesh modification
  /*! @internal Specifically injects the added and removed items into the total group
   *  of current all_items. Changes are made directly on this group without protection.
   */
  void finalizeMeshChanged();

  /*! \brief Largest local ID used.
    
    This number is used to allocate variable arrays
    on the mesh entities.
  */
  Integer maxUsedLocalId() const { return m_internals.size(); }

  //! Prepares variables for dumps
  void prepareForDump();

  //! Restores info from dumps
  void readFromDump();

  //! Group of all entities
  ItemGroup allItems() const { return m_all_group; }
  
  //! List of entities added or removed since the last endUpdate()
  Int32ConstArrayView addedItems  () const { return m_added_items;  }
  Int32ConstArrayView removedItems() const { return m_removed_items;}

  //! Internal list of ItemInternals
  /*! Read-only; the write version was removed as indicated in todo */
  ConstArrayView<ItemInternal*> itemsInternal() const { return m_internals; }

  ItemInternal* itemInternal(Int32 local_id) const { return m_internals[local_id]; }

  //! Adds an entity with a unique ID \a unique_id
  ItemInternal* allocOne(Int64 unique_id)
  {
    bool need_alloc = false;
    ItemInternal* next = _allocOne(need_alloc);
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyAdd(next,unique_id);
#endif
    if (m_has_unique_id_map)
      if (!m_items_map.add(unique_id,next))
        _badSameUniqueId(unique_id);
    return next;
  }

  //! Adds an entity with a unique ID \a unique_id
  ItemInternal* allocOne(Int64 unique_id,bool& need_alloc)
  {
    ItemInternal* next = _allocOne(need_alloc);
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyAdd(next,unique_id);
#endif
    if (m_has_unique_id_map)
      if (!m_items_map.add(unique_id,next))
        _badSameUniqueId(unique_id);
    return next;
  }

  //! Removes the entity \a item
  void removeOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    _setSuppressed(item);
    if (m_has_unique_id_map){
      Int64 uid = item->uniqueId().asInt64();
      if (uid != NULL_ITEM_UNIQUE_ID)
        m_items_map.remove(uid);
    }
    m_free_internals.add(item->localId());
    m_removed_items.add(item->localId());
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyRemove(item);
#endif
    --m_nb_item;
  }

  //! Removes the detached entity \a item
  void removeDetachedOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    _setSuppressed(item);
#ifndef REMOVE_UID_ON_DETACH
    if (m_has_unique_id_map)
      m_items_map.remove(item->uniqueId().asInt64());
#endif
    m_free_internals.add(item->localId());
    m_removed_items.add(item->localId());
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyRemove(item);
#endif
    --m_nb_item;
  }

  /*!
   * \brief Detaches the entity \a item.
   *
   * The entity is removed from the list of uniqueIds()
   * if the REMOVE_UID_ON_DETACH macro is defined
   */  
  void detachOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    // Maybe it should be marked suppressed.
    //_setSuppressed(item);
#ifdef REMOVE_UID_ON_DETACH
    if (m_has_unique_id_map)
      m_items_map.remove(item->uniqueId().asInt64());
#endif /* REMOVE_UID_ON_DETACH */
    item->setDetached(true);
  }

  //! Removes a list of entities
  void removeMany(Int32ConstArrayView local_ids);

  //! Finds the entity with unique ID \a unique_id and creates it if it does not exist
  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc)
  {
#ifdef ARCANE_CHECK
    if (!m_has_unique_id_map)
      _badUniqueIdMap();
#endif
    ItemInternalMap::LookupData item_data = m_items_map._lookupAdd(uid, 0, is_alloc);
    if (is_alloc){
      bool need_alloc;
      item_data.setValue(_allocOne(need_alloc));
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
      _notifyAdd(item_data.value(),uid);
#endif
    }
    return item_data.value();
  }

  //! Finds the entity with unique ID \a uid
  ItemInternal* findOne(Int64 uid)
  {
#ifdef ARCANE_CHECK
    if (!m_has_unique_id_map)
      _badUniqueIdMap();
#endif
    return m_items_map._tryFindItemInternal(uid);
  }

  //! Checks if the internal structures of the instance are valid
  void checkValid();

  ItemInternalMap& itemsMap() { return m_items_map; }

  Integer nbItem() const { return m_nb_item; }

  eItemKind kind() const { return m_kind; }

  bool changed() const
  { 
    return !m_added_items.empty() || !m_removed_items.empty();
  }

  void beginCompactItems(ItemFamilyCompactInfos& compact_infos);

  /*!
   * \brief Conversion between old and new local IDs.
   *
   * This method is only valid after calling beginCompactItems() and
   * before finishCompactItems().
   */
  ARCANE_DEPRECATED_240 Int32ConstArrayView oldToNewLocalIds() const;

  //! Removes all entities
  void clear();
  
  /*!
   * \brief Conversion between new and old local IDs.
   *
   * This method is only valid after calling beginCompactItems() and
   * before finishCompactItems().
   */
  ARCANE_DEPRECATED_240 Int32ConstArrayView newToOldLocalIds() const;

  void finishCompactItems(ItemFamilyCompactInfos& compact_infos);

  void itemsUniqueIdToLocalId(ArrayView<Int64> ids,bool do_fatal) const;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              Int64ConstArrayView unique_ids,
                              bool do_fatal) const;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              ConstArrayView<ItemUniqueId> unique_ids,
                              bool do_fatal) const;

  ItemFamily* itemFamily() const
  {
    return m_item_family;
  }

  void setItemFamily(ItemFamily* item_family);

  bool hasUniqueIdMap() const
  {
    return m_has_unique_id_map;
  }
  
  void setHasUniqueIdMap(bool v);

  void printFreeInternals(Integer max_print);

 public:

  EventObservableView<const ItemFamilyItemListChangedEventArgs&> itemListChangedEvent();

 private:

  /*! \brief Adds an entity.
   */
  ItemInternal* _allocOne(bool& need_alloc)
  {
    ItemInternal* new_item = 0;
    Integer nb_free = m_free_internals.size();
    Int32 lid = 0;
    if (nb_free!=0){
      new_item = m_internals[m_free_internals.back()];
      m_free_internals.popBack();
      _setAdded(new_item);
      lid = new_item->localId();
      need_alloc = false;
    }
    else{
      Integer nb_free2 = m_free_internals_in_multi_buffer.size();
      if (nb_free2!=0){
        new_item = m_free_internals_in_multi_buffer.back();
        m_free_internals_in_multi_buffer.popBack();
      }
      else
        new_item = m_item_internals_buffer->allocOne();
      lid = m_internals.size();
      new_item->setLocalId(lid);
      m_internals.add(new_item);
      _updateItemSharedInfoInternalView();
      need_alloc = true;
    }
    m_added_items.add(lid);
    ++m_nb_item;
    return new_item;
  }

 private:

  IMesh* m_mesh; //!< Associated mesh
  ItemFamily* m_item_family; //!< Associated mesh family
  eItemKind m_kind; //!< Corresponding kind
  String m_kind_name; //!< Name of the entity kind (Node, Cell, ...)
  String m_all_group_name; //!< Name of the group containing all entities
  ItemInternalMap m_items_map; //!< Hash table for uniqueId() -> ItemInternal* conversion
  ItemGroup m_all_group; //! Group of all entities
  Integer m_nb_item; //!< Number of allocated entities
  bool m_is_verbose;
  Int32UniqueArray m_added_items;
  Int32UniqueArray m_removed_items;
  bool m_use_new_finalize;
  bool m_is_first_finalize;
  bool m_has_unique_id_map;
  //! Temporary while oldToNewLocalIds() and newToOldLocalIds() exist
  ItemFamilyCompactInfos* m_compact_infos;
  ItemSharedInfo* m_common_item_shared_info = nullptr;
  EventObservable<const ItemFamilyItemListChangedEventArgs&> m_item_list_change_event;

 public:
  
  UniqueArray<ItemInternal*> m_internals; //!< ItemInternals of the entities
  Int32UniqueArray m_free_internals; //!< List of freed ItemInternals from m_internals
  //!< List of free ItemInternals from m_item_internals_buffer
  UniqueArray<ItemInternal*> m_free_internals_in_multi_buffer;

 private:

  /*! \brief Buffer to store an instance of ItemInternal.
   *
   * \warning A created instance must remain valid throughout
   * an execution.
   */
  MultiBufferT<ItemInternal>* m_item_internals_buffer;

 private:

  inline void _setSuppressed(ItemInternal* item)
  {
    int f = item->flags();
    f &= ~ItemFlags::II_Added;
    f |= ItemFlags::II_Suppressed;
    item->setFlags(f);
  }

  inline void _setAdded(ItemInternal* item)
  {
    int f = item->flags();
    f |= ItemFlags::II_Added;
    f &= ~ItemFlags::II_Suppressed;
    item->setFlags(f);
  }

  void _checkActiveItem(ItemInternal* item);
  void _dumpList();
  void _badSameUniqueId(Int64 unique_id) const;
  void _badUniqueIdMap() const;
  void _updateItemSharedInfoInternalView();
  void _notifyRemove(ItemInternal* item)
  {
    if (m_item_list_change_event.hasObservers())
      _notifyRemove2(item);
  }
  void _notifyAdd(ItemInternal* item,Int64 uid)
  {
    if (m_item_list_change_event.hasObservers())
      _notifyAdd2(item,uid);
  }
  void _notifyRemove2(ItemInternal* item);
  void _notifyAdd2(ItemInternal* item,Int64 uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
