// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemFamily.h                                             (C) 2000-2025 */
/*                                                                           */
/* Family of items of arbitrary types.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMFAMILY_H
#define ARCANE_CORE_ANYITEM_ANYITEMFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SharedPtr.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/anyitem/AnyItemGlobal.h"
#include "arcane/core/anyitem/AnyItemPrivate.h"
#include "arcane/core/anyitem/AnyItemGroup.h"
#include "arcane/core/anyitem/AnyItemFamilyObserver.h"
#include "arcane/core/ItemGroupObserver.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief AnyItem internal family
 * Aggregation of groups to describe variables / partial variables
 */
class FamilyInternal
{
 private:

  typedef std::set<IFamilyObserver*> FamilyObservers;

 public:

  FamilyInternal()
  : m_max_local_id(0)
  {}
  ~FamilyInternal()
  {
    clear();
  }

 public:

  //! Add a group to the family
  FamilyInternal& operator<<(GroupBuilder builder)
  {
    ItemGroup group = builder.group();
    const Integer size = m_groups.size();

    if (m_groups.findGroupInfo(group.internal()) != NULL)
      throw FatalErrorException(String::format("Group '{0}' already registered", group.name()));

    m_groups.resize(size + 1);
    Private::GroupIndexInfo& info = m_groups[size];
    info.group = group.internal();
    info.group_index = size;
    info.local_id_offset = m_max_local_id;
    m_max_local_id += group.size();
    info.is_partial = builder.isPartial();
    info.group->attachObserver(this, newItemGroupObserverT(this, &FamilyInternal::_notifyGroupHasChanged));
    // std::cout << "Attach " << this << " observer on group " << group.name() << "\n";

    // Notify observers
    _notifyFamilyIsIncreased();
    return *this;
  }

  //! Returns true if the family contains the group
  inline bool contains(const ItemGroup& group) const
  {
    return (m_groups.findGroupInfo(group.internal()) != NULL);
  }

  //! Returns true if the group is associated with a partial variable
  inline bool isPartial(const ItemGroup& group) const
  {
    const Private::GroupIndexInfo* info = m_groups.findGroupInfo(group.internal());
    if (info == NULL)
      throw Arcane::FatalErrorException(Arcane::String::format("Group '{0}' not registered", group.name()));
    return info->is_partial;
  }

  //! Group of all items
  // This group does not need to observe the family because it shares the data
  inline Group allItems() const
  {
    return m_groups;
  }

  //! Position of the group in the family
  inline Integer groupIndex(const ItemGroup& group) const
  {
    const Private::GroupIndexInfo* info = m_groups.findGroupInfo(group.internal());
    if (info == NULL)
      throw FatalErrorException(String::format("Group '{0}' not registered", group.name()));
    return info->group_index;
  }

  //! Position in the family of the first localId of this group
  inline Integer firstLocalId(const ItemGroup& group) const
  {
    const Private::GroupIndexInfo* info = m_groups.findGroupInfo(group.internal());
    if (!info)
      ARCANE_FATAL("Group '{0}' not registered", group.name());
    return info->local_id_offset;
  }

  //! Returns the concrete item associated with this AnyItem
  template <typename AnyItemT>
  Item item(const AnyItemT& any_item) const
  {
    // NOTE GG: the value of group.itemInfoListView() does not change during
    // calculation, so it is possible to keep it as a class field.
    const Integer group_index = any_item.groupIndex();
    const Private::GroupIndexInfo& info = m_groups[group_index];
    const ItemGroupImpl& group = *(info.group);
    Integer index_in_group = any_item.localId() - info.local_id_offset;
    Item item = group.itemInfoListView()[group.itemsLocalId()[index_in_group]];
    // ARCANE_ASSERT((!info.is_partial || (item->localId() == any_item.varIndex())),("Inconsistent concrete item"));
    // ARCANE_ASSERT((item->isOwn() == any_item.m_is_own),("Inconsistent concrete item isOwn"));
    return item;
  }

  //! Size of the family, i.e., number of groups
  inline Integer groupSize() const
  {
    return m_groups.size();
  }

  //! Number of items in this family
  /*! Sum of the size of all groups composing it */
  inline Integer maxLocalId() const
  {
    return m_max_local_id;
  }

  //! Accessor for the i-th group of the family
  ItemGroup group(Integer i) const
  {
    return ItemGroup(m_groups[i].group);
  }

  //! Clear the family
  void clear()
  {
    for (Integer igrp = 0; igrp < m_groups.size(); ++igrp) {
      m_groups[igrp].group->detachObserver(this);
      // std::cout << "Detach " << this << " observer on group " << m_groups[igrp].group->name() << "\n";
    }
    // Clear
    m_groups.clear();
    m_max_local_id = 0;
    // Notify observers
    _notifyFamilyIsInvalidate();
  }

  //! Register an observer
  void registerObserver(IFamilyObserver& observer) const
  {
    FamilyObservers::const_iterator it = m_observers.find(&observer);
    if (it != m_observers.end())
      throw FatalErrorException("FamilyObserver already registered");
    m_observers.insert(&observer);
  }

  //! Remove an observer
  void removeObserver(IFamilyObserver& observer) const
  {
    FamilyObservers::const_iterator it = m_observers.find(&observer);
    if (it == m_observers.end())
      throw FatalErrorException("FamilyObserver not registered");
    m_observers.erase(it);
  }

 public:

  const Private::GroupIndexInfo* findGroupInfo(ItemGroup agroup)
  {
    return m_groups.findGroupInfo(agroup.internal());
  }

 private:

  void _notifyFamilyIsInvalidate()
  {
    for (FamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsInvalidate();
  }

  void _notifyFamilyIsIncreased()
  {
    for (FamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsIncreased();
  }

  void _notifyGroupHasChanged()
  {
    throw FatalErrorException(A_FUNCINFO, "Group changes while registered in AnyItem::Family");
  }

 private:

  //! Container of groups
  Private::GroupIndexMapping m_groups;

  //! Maximum identifier (equivalent to the size of the family)
  Integer m_max_local_id;

  //! So that objects built on the family cannot modify it
  mutable FamilyObservers m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief AnyItem family (flyweight pattern)
 * Aggregation of groups to describe variables / partial variables
 * Copy by reference
 */
class Family
{
 public:

  Family()
  : m_internal(new FamilyInternal)
  {}
  Family(const Family& f)
  : m_internal(f.m_internal)
  {}
  ~Family() {}

 public:

  //! Comparisons
  bool operator==(const Family& f) const { return m_internal == f.m_internal; }
  bool operator!=(const Family& f) const { return !operator==(f); }

  Family& operator=(const Family& f)
  {
    m_internal = f.m_internal;
    return *this;
  }

  //! Add a group to the family
  Family& operator<<(GroupBuilder builder)
  {
    *m_internal << builder;
    return *this;
  }

  //! Returns true if the family contains the group
  inline bool contains(const ItemGroup& group) const
  {
    return m_internal->contains(group);
  }

  //! Returns true if the group is associated with a partial variable
  inline bool isPartial(const ItemGroup& group) const
  {
    return m_internal->isPartial(group);
  }

  //! Group of all items
  inline Group allItems()
  {
    return m_internal->allItems();
  }

  //! Position of the group in the family
  inline Integer groupIndex(const ItemGroup& group) const
  {
    return m_internal->groupIndex(group);
  }

  //! Position in the family of the first localId of this group
  inline Integer firstLocalId(const ItemGroup& group) const
  {
    return m_internal->firstLocalId(group);
  }

  //! Returns the concrete item associated with this AnyItem
  template <typename AnyItemT>
  Item item(const AnyItemT& any_item) const
  {
    return m_internal->item(any_item);
  }

  //! Size of the family, i.e., number of groups
  inline Integer groupSize() const
  {
    return m_internal->groupSize();
  }

  //! Number of items in this family
  /*! Sum of the size of all groups composing it */
  inline Integer maxLocalId() const
  {
    return m_internal->maxLocalId();
  }

  //! Accessor for the i-th group of the family
  ItemGroup group(Integer i) const
  {
    return m_internal->group(i);
  }

  //! Clear the family
  void clear()
  {
    m_internal->clear();
  }

  //! Register an observer
  void registerObserver(IFamilyObserver& observer) const
  {
    m_internal->registerObserver(observer);
  }

  //! Remove an observer
  void removeObserver(IFamilyObserver& observer) const
  {
    m_internal->removeObserver(observer);
  }

  FamilyInternal* internal() const
  {
    return m_internal.get();
  }

 private:

  //! Internal family
  SharedPtrT<FamilyInternal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
