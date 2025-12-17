// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemUserGroup.h                                          (C) 2000-2025 */
/*                                                                           */
/* Groupe utilisateur aggrégée de types quelconques.                         */
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
 * \brief Groupe utilisateur
 * pour spécifier des groupes (Arcane) sur lesquels on souhaite itérer
 * ces groupes doivent être dans la famille
 *
 * Par exemple :
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

  //! Ajout d'un groupe arcane au groupe
  inline UserGroup& operator<<(GroupBuilder builder)
  {
    ItemGroup group = builder.group();
    if (m_groups.findGroupInfo(group.internal()) != NULL)
      throw FatalErrorException(String::format("Group '{0}' in user group already registered",group.name()));

    const Private::GroupIndexInfo * info = m_family.internal()->findGroupInfo(group);
    if (info == NULL)
      throw FatalErrorException(String::format("Group '{0}' in user group not registered in family",group.name()));

    if(builder.isPartial() != info->is_partial)
      throw FatalErrorException(String::format("Group '{0}' in user group is not same in family",group.name()));
    
    m_currents.add(*info);
    return *this;
  }
  
  //! Vide le groupe
  inline void clear() {
    m_currents.clear();
  }

  //! Action si la famille est invalidée : on vide le groupe
  inline void notifyFamilyIsInvalidate() {
    // Si la famille change, on invalide le groupe
    clear();
  }

  //! Si la famille est agrandie, pas d'impact sur le groupe
  inline void notifyFamilyIsIncreased() {
    // On ne fait rien dans ce cas
  }
  
private:
  
  //! Famille AnyItem (copie flyweight)
  const Family m_family;
  
  //! Table Groupe - offset
  GroupIndexMapping m_currents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMUSERGROUP_H */
