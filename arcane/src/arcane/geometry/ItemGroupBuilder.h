// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ITEMGROUPBUILDER_H
#define ITEMGROUPBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <set>
#include <cstring>
#include <cctype>

#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/ArcaneVersion.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroupRangeIterator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

//! Macro for object name construction
/*! Generally used to name groups for ItemGroupBuilder */
#define IMPLICIT_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__), false)
#define IMPLICIT_UNIQ_NAME ItemGroupBuilder_cleanString(__FILE__ "__" TOSTRING(__LINE__), true)

/*
 * \internal
 * \brief Assisted group building tool
 *
 * The uniqueness of the group elements is guaranteed by construction. It
 * is possible to use the IMPLICIT_NAME macro to name the group.
 */
template <typename T>
class ItemGroupBuilder
{
 private:

  IMesh* m_mesh;
  std::set<Integer> m_ids;
  String m_group_name;

 public:

  //! Constructor
  ItemGroupBuilder(IMesh* mesh, const String& groupName)
  : m_mesh(mesh)
  , m_group_name(groupName)
  {}

  //! Destructor
  virtual ~ItemGroupBuilder() {}

 public:

  //! Add a set of items provided by an enumerator
  void add(ItemEnumeratorT<T> enumerator)
  {
    while (enumerator.hasNext()) {
      m_ids.insert(enumerator.localId());
      ++enumerator;
    }
  }

  //! Add a set of items provided by an enumerator
  void add(ItemGroupRangeIteratorT<T> enumerator)
  {
    while (enumerator.hasNext()) {
      m_ids.insert(enumerator.itemLocalId());
      ++enumerator;
    }
  }

  //! Add a unique item
  void add(const T& item)
  {
    m_ids.insert(item.localId());
  }

  //! Constructor for the new group
  ItemGroupT<T> buildGroup()
  {
    Int32UniqueArray localIds(m_ids.size());

    std::set<Integer>::const_iterator is = m_ids.begin();
    Integer i = 0;

    while (is != m_ids.end()) {
      localIds[i] = *is;
      ++is;
      ++i;
    }

    //       ItemGroup newGroup(new ItemGroupImpl(m_mesh->itemFamily(ItemTraitsT<T>::kind()),
    //                                            m_group_name));
    ItemGroup newGroup = m_mesh->itemFamily(ItemTraitsT<T>::kind())->findGroup(m_group_name, true);
    // m_item_family->createGroup(own_name,ItemGroup(this));

    newGroup.clear();
    newGroup.setItems(localIds);
    // newGroup.setLocalToSubDomain(true); // Forces the new group to be local: do not transfer in case of rebalancing

    return newGroup;
  }

  //! Group name
  String getName() const
  {
    return m_group_name;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
