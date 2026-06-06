// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemGroup.h                                              (C) 2000-2025 */
/*                                                                           */
/* Aggregated group of arbitrary types.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMGROUP_H
#define ARCANE_CORE_ANYITEM_ANYITEMGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/anyitem/AnyItemGlobal.h"
#include "arcane/core/anyitem/AnyItemPrivate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NB: It must be known very early whether we are iterating over a variable
 * or a partial variable
 *
 */

/*!
 * \brief Tool for building a group
 */
class GroupBuilder
{
public:
  GroupBuilder(ItemGroup g) 
    : m_group(g)
    , m_is_partial(false) {}
  ItemGroup group() const { return m_group; }
  bool isPartial() const { return m_is_partial; }
protected:  
  ItemGroup m_group;
  bool m_is_partial;
};

/*!
 * \brief Tool for building a group for a partial variable
 */
class PartialGroupBuilder 
  : public GroupBuilder
{
public:
  PartialGroupBuilder(ItemGroup g) : GroupBuilder(g) {
    this->m_is_partial = true;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief AnyItem Group
 * Aggregation of Arcane group + information {partial or not} for variables
 * Construction within AnyItem families
 */
class Group
{
  static ItemInternal* _toInternal(const Item& v)
  {
    return ItemCompatibility::_itemInternal(v);
  }

public:

  /*!
   * \brief Enumerator of an item block
   *
   * Arcane enumerator enriched with the position in the family
   *
   */
  class BlockItemEnumerator
  {
  private:
    typedef ItemInternal* ItemInternalPtr;

  public:
    BlockItemEnumerator(const Private::GroupIndexInfo & info)
      : m_info(info)
      , m_items(m_info.group->itemInfoListView()), m_local_ids(m_info.group->itemsLocalId().data())
      , m_index(0), m_count(m_info.group->size()), m_is_partial(info.is_partial) { }

    BlockItemEnumerator(const BlockItemEnumerator& e) 
      : m_info(e.m_info)
      , m_items(e.m_items), m_local_ids(e.m_local_ids)
      , m_index(e.m_index), m_count(e.m_count), m_is_partial(e.m_is_partial) {}

    //! Dereference to the associated Arcane item
    Item operator*() const { return m_items[ m_local_ids[m_index] ]; }
    // TODO: return an 'Item*' similar to ItemEnumerator. 
    //! Indirect dereference to the associated Arcane item
    ItemInternal* operator->() const { return Group::_toInternal(m_items[ m_local_ids[m_index] ]); }
    //! Advancement of the enumerator
    inline void operator++() { ++m_index; }
    //! Test for end of enumerator
    inline bool hasNext() { return m_index<m_count; }
    //! Number of elements in the enumerator
    inline Integer count() const { return m_count; }
    
    //! localId() of the current entity.
    inline Integer varIndex() const { return (m_is_partial)?m_index:m_local_ids[m_index]; }
    
    //! localId() of the current entity.
    inline Integer localId() const { return m_info.local_id_offset+m_index; }

    //! Index in the current AnyItem::Family group
    inline Integer groupIndex() const { return m_info.group_index; }

    //! Current underlying group
    inline ItemGroup group() const { return ItemGroup(m_info.group); }
    
  private:
    const Private::GroupIndexInfo & m_info;

    ItemInfoListView m_items;
    const Int32* ARCANE_RESTRICT m_local_ids;
    Integer m_index;
    Integer m_count;
    bool m_is_partial;
  };
  
  /*!
   * \brief Enumerator of item blocks
   */
  class Enumerator
  {
  public:
    Enumerator(const Private::GroupIndexMapping& groups) 
    : m_current(std::begin(groups))
      , m_end(std::end(groups)) {}
    Enumerator(const Enumerator& e) 
      : m_current(e.m_current)
      , m_end(e.m_end) {}
    inline bool hasNext() const { return m_current != m_end; }
    inline void operator++() { m_current++; }
    //! Enumerator of an item block
    inline BlockItemEnumerator enumerator() {
      return BlockItemEnumerator(*m_current);
    }
    inline Integer groupIndex() const { return m_current->group_index; }
    ItemGroup group() const { return ItemGroup(m_current->group); }
  private:
    Private::GroupIndexMapping::const_iterator m_current;
    Private::GroupIndexMapping::const_iterator m_end;
  };
  
public:

  //! Construction from a Group - offset table (from the family)
  Group(const Private::GroupIndexMapping& groups) 
    : m_groups(groups) {} 
  
  //! Enumerator of the group
  inline Enumerator enumerator() const {
    return Enumerator(m_groups);
  }
  
  //! Number of aggregated groups
  inline Integer size() const { 
    return m_groups.size();
  }

  //private:
public:
  
  //! Group - offset table
  const Private::GroupIndexMapping& m_groups;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
     
#endif /* ARCANE_ANYITEM_ANYITEMGROUP_H */
