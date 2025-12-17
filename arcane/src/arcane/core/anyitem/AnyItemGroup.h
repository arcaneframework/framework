// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemGroup.h                                              (C) 2000-2025 */
/*                                                                           */
/* Groupe aggrégée de types quelconques.                                     */
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
 * NB: Il faut savoir très tôt si on va itérer pour une variable ou une variable partielle
 *
 */

/*!
 * \brief Outil pour construire un groupe
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
 * \brief Outil pour construire un groupe pour une variable partielle
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
 * \brief Groupe AnyItem
 * Agglomération de groupe Arcane + informations {partiel ou non} pour les variables
 * Construction dans les familles AnyItem
 */
class Group
{
  static ItemInternal* _toInternal(const Item& v)
  {
    return ItemCompatibility::_itemInternal(v);
  }

public:

  /*!
   * \brief Enumérateur d'un bloc d'items
   *
   * Enumérateur Arcane enrichi de la position dans la famille
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

    //! Déréférencement vers l'item Arcane associé
    Item operator*() const { return m_items[ m_local_ids[m_index] ]; }
    // TODO: retourner un 'Item*' de manière similaire à ItemEnumerator. 
    //! Déréférencement indirect vers l'item Arcane associé
    ItemInternal* operator->() const { return Group::_toInternal(m_items[ m_local_ids[m_index] ]); }
    //! Avancement de l'énumérateur
    inline void operator++() { ++m_index; }
    //! Test de fin de l'énumérateur
    inline bool hasNext() { return m_index<m_count; }
    //! Nombre d'éléments de l'énumérateur
    inline Integer count() const { return m_count; }
    
    //! localId() de l'entité courante.
    inline Integer varIndex() const { return (m_is_partial)?m_index:m_local_ids[m_index]; }
    
    //! localId() de l'entité courante.
    inline Integer localId() const { return m_info.local_id_offset+m_index; }

    //! Index dans la AnyItem::Family du groupe en cours
    inline Integer groupIndex() const { return m_info.group_index; }

    //! Groupe sous-jacent courant
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
   * \brief Enumérateur des blocs d'items
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
    //! Enumérateur d'un bloc d'items
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

  //! Construction à partir d'une table Groupe - offset (issue de la famille)
  Group(const Private::GroupIndexMapping& groups) 
    : m_groups(groups) {} 
  
  //! Enumérateur du groupe
  inline Enumerator enumerator() const {
    return Enumerator(m_groups);
  }
  
  //! Nombre de groupes aggrégés
  inline Integer size() const { 
    return m_groups.size();
  }

  //private:
public:
  
  //! Table Groupe - offset
  const Private::GroupIndexMapping& m_groups;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
     
#endif /* ARCANE_ANYITEM_ANYITEMGROUP_H */
