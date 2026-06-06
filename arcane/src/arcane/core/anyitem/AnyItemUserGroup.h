// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemUserGroup.h                                          (C) 2000-2025 */
/*                                                                           */
/* Aggregated user group of arbitrary types.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMUSERGROUP_H
#define ARCANE_CORE_ANYITEM_ANYITEMUSERGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

#include "arcane/core/anyitem/AnyItemGlobal.h"
#include "arcane/core/anyitem/AnyItemPrivate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief User group
 * to specify groups (Arcane) over which one wishes to iterate
 * these groups must be in the family
 *
 * For example:
 *
 * AnyItem::Family family;
 *
 * family << AnyItem::GroupBuilder( allFaces() ) 
 *        << AnyItem::GroupBuilder( allCells() )
 *        << AnyItem::GroupBuilder( allNodes() );
 *
 * AnyItem::UserGroup sub_group(family);
 *
 * sub_group << AnyItem::GroupBuilder( allCells() )
 *           << AnyItem::GroupBuilder( allFaces() );
 */
class UserGroup
: public Group
, public IFamilyObserver
{
 private:

  typedef Private::GroupIndexMapping GroupIndexMapping;

 public:

  UserGroup(const Family& family)
  : Group(m_currents)
  , m_family(family)
  {
    m_family.registerObserver(*this);
  }

  ~UserGroup()
  {
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this); });
  }

  //! Adds an arcane group to the group
  inline UserGroup& operator<<(GroupBuilder builder)
  {
    ItemGroup group = builder.group();
    if (m_groups.findGroupInfo(group.internal()) != NULL)
      throw FatalErrorException(String::format("Group '{0}' in user group already registered", group.name()));

    const Private::GroupIndexInfo* info = m_family.internal()->findGroupInfo(group);
    if (info == NULL)
      throw FatalErrorException(String::format("Group '{0}' in user group not registered in family", group.name()));

    if (builder.isPartial() != info->is_partial)
      throw FatalErrorException(String::format("Group '{0}' in user group is not same in family", group.name()));

    m_currents.add(*info);
    return *this;
  }

  //! Clears the group
  inline void clear()
  {
    m_currents.clear();
  }

  //! Action if the family is invalidated: clears the group
  inline void notifyFamilyIsInvalidate()
  {
    // If the family changes, the group is invalidated
    clear();
  }

  //! If the family is increased, no impact on the group
  inline void notifyFamilyIsIncreased()
  {
    // Nothing is done in this case
  }

 private:

  //! AnyItem Family (flyweight copy)
  const Family m_family;

  //! Group Table - offset
  GroupIndexMapping m_currents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
