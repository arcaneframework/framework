// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupInternal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Internal part of ItemGroup.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ITEMGROUPINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITEMGROUPINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"

#include <map>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of sub-parts of a group based on the type of its elements.
 *
 * This class allows retrieving the sub-part of a group corresponding
 * to a basic type (entity whose ItemTypeId is less than NB_BASIC_ITEM_TYPE.
 *
 * There are two implementations of this functionality.
 *
 * The first version, which is obsolete and less used by default,
 * uses an ItemGroup per entity type. The second uses just an array
 * of ItemLocalId for each part. Furthermore, if all elements of the group
 * are of the same type (for example, an IT_Quad4 if using a 2D Cartesian mesh),
 * then the list of localId() from the instance is used directly.
 */
class ItemGroupSubPartsByType
{
 public:

  explicit ItemGroupSubPartsByType(ItemGroupInternal* igi);

 public:

  void setImpl(ItemGroupImpl* group_impl) { m_group_impl = group_impl; }
  void clear()
  {
    m_children_by_type.clear();
    m_children_by_type_ids.clear();
  }
  void applyOperation(IItemOperationByBasicType* operation);
  bool isUseV2ForApplyOperation() const { return m_use_v2_for_apply_operation; }

  void _computeChildrenByTypeV1();

 private:

  void _initChildrenByTypeV2();
  void _computeChildrenByTypeV2();
  void _initChildrenByTypeV1();

 private:

  //! True if version 2 of the management is used for applyOperation().
  bool m_use_v2_for_apply_operation = true;

  /*!
   * \brief List of localIds by entity type.
   *
   * This field is used with version 2.
   */
  UniqueArray<UniqueArray<Int32>> m_children_by_type_ids;

  /*!
   * \brief List of children of this group by entity type.
   *
   * This field is used with version 1, which requires
   * creating a group per sub-type.
   */
  UniqueArray<ItemGroupImpl*> m_children_by_type;

  /*!
   * \brief Indicates the type of the entities in the group.
   *
   * If different from IT_NullType, it means that all
   * entities in the group are of the same type and therefore it is not
   * necessary to calculate the localId() of the entities by type.
   * In this case, the group passed to applyOperation() is used directly.
   */
  ItemTypeId m_unique_children_type{ IT_NullType };

  //! Timestamp indicating when the list of child ids was calculated
  Int64 m_children_by_type_ids_computed_timestamp = -1;

  bool m_is_debug_apply_operation = false;

  ItemGroupInternal* m_group_internal = nullptr;

  //! To be deleted when version V1 is removed
  ItemGroupImpl* m_group_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implementation of the ItemGroupImpl class.
 *
 * The container holding the list of group entities is either a
 * variable in the case of a standard group, or a simple array
 * in the case of a group having a parent. Indeed, groups
 * having parents are dynamically generated groups (for
 * example, the group of own entities) and are therefore not
 * always present in all sub-domains (a variable must always
 * exist in all sub-domains). Furthermore, their value does not need to be
 * saved during a protection.

 \todo add notion of generated group, with the following properties:
 - these groups must not be transferred from one subdomain to another
 - they cannot be modified directly.
 */
class ItemGroupInternal
{
  friend class ItemGroupImplInternal;

 public:

  /*!
   * \brief Mutex to protect calls to ItemGroupImpl::_checkNeedUpdate().
   *
   * By default, the mutex is not active. You must call create() to
   * activate it.
   */
  class CheckNeedUpdateMutex
  {
   public:

    class ScopedLock
    {
     public:

      explicit ScopedLock(const CheckNeedUpdateMutex& mutex)
      : m_update_mutex(mutex)
      {
        m_update_mutex._lock();
      }
      ~ScopedLock()
      {
        m_update_mutex._unlock();
      }

     private:

      const CheckNeedUpdateMutex& m_update_mutex;
    };

   public:

    ~CheckNeedUpdateMutex()
    {
      delete m_mutex;
    }
    void create()
    {
      m_mutex = new std::mutex();
    }

   private:

    std::mutex* m_mutex = nullptr;

   private:

    void _lock() const
    {
      if (m_mutex)
        m_mutex->lock();
    }
    void _unlock() const
    {
      if (m_mutex)
        m_mutex->unlock();
    }
  };

 public:

  ItemGroupInternal();
  ItemGroupInternal(IItemFamily* family, const String& name);
  ItemGroupInternal(IItemFamily* family, ItemGroupImpl* parent, const String& name);
  ~ItemGroupInternal();

 public:

  const String& name() const { return m_name; }
  const String& fullName() const { return m_full_name; }
  bool null() const { return m_is_null; }
  IMesh* mesh() const { return m_mesh; }
  eItemKind kind() const { return m_kind; }
  Integer maxLocalId() const;
  ItemInternalList items() const;
  ItemInfoListView itemInfoListView() const;

  Int32ArrayView itemsLocalId() { return *m_items_local_id; }
  Int32ConstArrayView itemsLocalId() const { return *m_items_local_id; }
  Int32Array& mutableItemsLocalId() { return *m_items_local_id; }
  VariableArrayInt32* variableItemsLocalid() { return m_variable_items_local_id; }

  Int64 timestamp() const { return m_timestamp; }
  bool isContiguous() const { return m_is_contiguous; }
  void checkIsContiguous();

  void updateTimestamp()
  {
    ++m_timestamp;
    m_is_contiguous = false;
  }

  void setNeedRecompute()
  {
    // NOTE: normally, this value should only be set to 'true' for
    // recalculated groups (which have a parent or for which 'm_compute_functor' is
    // not null). However, this method is also called on the group of all entities
    // and perhaps other groups.
    // Changing this behavior risks impacting a lot of code, so it should be
    // properly verified before making this modification.
    m_need_recompute = true;
  }

  //! Apply padding for vectorization
  void applySimdPadding();

  void checkUpdateSimdPadding();
  bool isAllItems() const { return m_is_all_items; }
  bool isOwn() const { return m_is_own; }
  Int32 nbItem() const { return itemsLocalId().size(); }
  void checkValid();

 public:

  void _removeItems(SmallSpan<const Int32> items_local_id);

 private:

  void _notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining);

 public:

  ItemGroupImplInternal m_internal_api;
  IMesh* m_mesh = nullptr; //!< Associated group manager
  IItemFamily* m_item_family = nullptr; //!< Associated family
  ItemGroupImpl* m_parent = nullptr; //! Parent group (null group if none)
  String m_variable_name; //!< Name of the variable containing the group item indices
  String m_full_name; //!< Full name of the group.
  bool m_is_null = true; //!< \a true if the group is null
  eItemKind m_kind = IK_Unknown; //!< Kind of the group entities
  String m_name; //!< Name of the group
  bool m_is_own = false; //!< \a true if the group contains only entities owned by us.

 private:

  Int64 m_timestamp = -1; //!< Time of the last modification

 public:

  Int64 m_simd_timestamp = -1; //!< Time of the last modification for SIMD info calculation
  //@{ @name local aliases to sub-groups of m_sub_groups
  ItemGroupImpl* m_own_group = nullptr; //!< Items owned by the subdomain
  ItemGroupImpl* m_ghost_group = nullptr; //!< Items not owned by the subdomain
  ItemGroupImpl* m_interface_group = nullptr; //!< Items on the boundary of two subdomains
  ItemGroupImpl* m_node_group = nullptr; //!< Node group
  ItemGroupImpl* m_edge_group = nullptr; //!< Edge group
  ItemGroupImpl* m_face_group = nullptr; //!< Face group
  ItemGroupImpl* m_cell_group = nullptr; //!< Mesh group
  ItemGroupImpl* m_inner_face_group = nullptr; //!< Inner face group
  ItemGroupImpl* m_outer_face_group = nullptr; //!< Outer face group
  //! AMR
  // FIXME we can avoid storing these groups by introducing predicates
  // on the parent groups
  ItemGroupImpl* m_active_cell_group = nullptr; //!< Active mesh group
  ItemGroupImpl* m_own_active_cell_group = nullptr; //!< Active owned mesh group
  ItemGroupImpl* m_active_face_group = nullptr; //!< Active face group
  ItemGroupImpl* m_own_active_face_group = nullptr; //!< Active owned face group
  ItemGroupImpl* m_inner_active_face_group = nullptr; //!< Active inner face group
  ItemGroupImpl* m_outer_active_face_group = nullptr; //!< Active outer face group
  std::map<Integer, ItemGroupImpl*> m_level_cell_group; //!< Level mesh group
  std::map<Integer, ItemGroupImpl*> m_own_level_cell_group; //!< Level owned mesh group

  //@}
  std::map<String, AutoRefT<ItemGroupImpl>> m_sub_groups; //!< Set of all sub-groups
  bool m_need_recompute = false; //!< True if the group needs to be recalculated
  bool m_need_invalidate_on_recompute = false; //!< True if invalidate observers must be activated upon recalculation
  bool m_transaction_mode = false; //!< True if the group is in direct transaction mode
  bool m_is_local_to_sub_domain = false; //!< True if the group is local to the subdomain
  IFunctor* m_compute_functor = nullptr; //!< Group computation function
  bool m_is_all_items = false; //!< Indicates if it is the group of all entities
  bool m_is_constituent_group = false; //!< Indicates if the group is associated with a constituent (IMeshComponent)
  SharedPtrT<GroupIndexTable> m_group_index_table; //!< Hash table of item local id to its enumeration position
  Ref<IVariableSynchronizer> m_synchronizer; //!< Group synchronizer

  // Previously in DynamicMeshKindInfo
  UniqueArray<Int32> m_items_index_in_all_group; //! localids -> index (ONLY ALLITEMS)

  std::map<const void*, IItemGroupObserver*> m_observers; //!< Group observers
  bool m_observer_need_info = false; //!< Synthesis of observer need for transition information
  void notifyExtendObservers(const Int32ConstArrayView* info);
  void notifyReduceObservers(const Int32ConstArrayView* info);
  void notifyCompactObservers(const Int32ConstArrayView* info);
  void notifyInvalidateObservers();

  void resetSubGroups();

 public:

  UniqueArray<Int32> m_local_buffer{ MemoryUtils::getAllocatorForMostlyReadOnlyData() };
  Array<Int32>* m_items_local_id = &m_local_buffer; //!< List of local IDs of the entities in this group
  VariableArrayInt32* m_variable_items_local_id = nullptr;
  bool m_is_contiguous = false; //! True if the localIds are consecutive.
  bool m_is_check_simd_padding = true;
  bool m_is_print_check_simd_padding = false;
  bool m_is_print_apply_simd_padding = false;
  bool m_is_print_stack_apply_simd_padding = false;

 public:

  //! Mutex to protect updates.
  CheckNeedUpdateMutex m_check_need_update_mutex;

 public:

  //! Sub-part of a group based on its type
  ItemGroupSubPartsByType m_sub_parts_by_type;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
