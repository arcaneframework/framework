// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemFamily.h                                             (C) 2000-2023 */
/*                                                                           */
/* Famille d'items de types quelconques.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ANYITEM_ANYITEMFAMILY_H
#define ARCANE_ANYITEM_ANYITEMFAMILY_H 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/SharedPtr.h>

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
 * \brief Famille AnyItem partie interne
 * Aggrégation de groupes pour décrire des variables / variables partielles
 */
class FamilyInternal
{ 
private:
  typedef std::set<IFamilyObserver*> FamilyObservers;
  
public:
  
  FamilyInternal() : m_max_local_id(0) {}
  ~FamilyInternal() 
  {
    clear();
  }
  
public:
  
  //! Ajout d'un groupe dans la famille
  FamilyInternal& operator<<(GroupBuilder builder)
  {
    ItemGroup group = builder.group();
    const Integer size = m_groups.size();

    if (m_groups.findGroupInfo(group.internal()) != NULL)
      throw FatalErrorException(String::format("Group '{0}' already registered",group.name()));    
    
    m_groups.resize(size+1);
    Private::GroupIndexInfo & info = m_groups[size];
    info.group = group.internal();
    info.group_index = size;
    info.local_id_offset = m_max_local_id;
    m_max_local_id += group.size();
    info.is_partial = builder.isPartial();
    info.group->attachObserver(this, newItemGroupObserverT(this, &FamilyInternal::_notifyGroupHasChanged));
    // std::cout << "Attach " << this << " observer on group " << group.name() << "\n";

    // On previent les observeurs
    _notifyFamilyIsIncreased();
    return *this;
  }

  //! retroune vrai si la famille contient le groupe
  inline bool contains(const ItemGroup& group) const 
  {
    return (m_groups.findGroupInfo(group.internal()) != NULL);
  }
  
  //! retourne vrai si le groupe est associé à une variable partielle
  inline bool isPartial(const ItemGroup& group) const 
  {
    const Private::GroupIndexInfo * info = m_groups.findGroupInfo(group.internal());
    if (info == NULL)
      throw Arcane::FatalErrorException(Arcane::String::format("Group '{0}' not registered",group.name()));
    return info->is_partial;
  }
  
  //! Groupe de tous les items
  // Ce groupe n'a pas besoin d'observer la famille car il partage les données
  inline Group allItems() const {
    return m_groups;
  }
  
  //! Position du groupe dans la famille
  inline Integer groupIndex(const ItemGroup& group) const {
    const Private::GroupIndexInfo * info = m_groups.findGroupInfo(group.internal());
    if (info == NULL)
      throw FatalErrorException(String::format("Group '{0}' not registered",group.name()));
    return info->group_index;
  }

  //! Position dans la famille du premier localId de ce groupe
  inline Integer firstLocalId(const ItemGroup& group) const
  {
    const Private::GroupIndexInfo * info = m_groups.findGroupInfo(group.internal());
    if (!info)
      ARCANE_FATAL("Group '{0}' not registered",group.name());
    return info->local_id_offset;
  }
  
  //! Retoune l'item concret associé à ce AnyItem
  template<typename AnyItemT>
  Item item(const AnyItemT & any_item) const
  {
    // NOTE GG: la valeur de group.itemInfoListView() ne change pas au cours
    // du calcul donc il est possible de la conserver comme champ de la classe.
    const Integer group_index = any_item.groupIndex();
    const Private::GroupIndexInfo & info = m_groups[group_index];
    const ItemGroupImpl & group = *(info.group);
    Integer index_in_group = any_item.localId() - info.local_id_offset;
    Item item = group.itemInfoListView()[group.itemsLocalId()[index_in_group]];
    // ARCANE_ASSERT((!info.is_partial || (item->localId() == any_item.varIndex())),("Inconsistent concrete item"));
    // ARCANE_ASSERT((item->isOwn() == any_item.m_is_own),("Inconsistent concrete item isOwn"));
    return item;
  }

  //! Taille de la famille, ie nombre de groupes
  inline Integer groupSize() const { 
    return m_groups.size();
  }
  
  //! Nombre d'items dans cette famille
  /*! Somme de la taille de tous les groupes la composant */
  inline Integer maxLocalId() const {
    return m_max_local_id;
  }

  //! Accesseur au i-ème groupe de la famille
  ItemGroup group(Integer i) const {
    return ItemGroup(m_groups[i].group);
  }

  //! Vide la famille
  void clear() {
    for(Integer igrp=0;igrp<m_groups.size();++igrp) {
      m_groups[igrp].group->detachObserver(this);
      // std::cout << "Detach " << this << " observer on group " << m_groups[igrp].group->name() << "\n";
    }
    // On efface
    m_groups.clear();
    m_max_local_id = 0;
    // On previent les observeurs
    _notifyFamilyIsInvalidate();
  }

  //! Enregistre un observeur
  void registerObserver(IFamilyObserver& observer) const
  {
    FamilyObservers::const_iterator it = m_observers.find(&observer);
    if(it != m_observers.end())
      throw FatalErrorException("FamilyObserver already registered");
    m_observers.insert(&observer);
  }
  
  //! Supprime un observeur
  void removeObserver(IFamilyObserver& observer) const
  {
    FamilyObservers::const_iterator it = m_observers.find(&observer);
    if(it == m_observers.end())
      throw FatalErrorException("FamilyObserver not registered");
    m_observers.erase(it);
  }

public:
  const Private::GroupIndexInfo * findGroupInfo(ItemGroup agroup) {
    return m_groups.findGroupInfo(agroup.internal());
  }

private:

  void _notifyFamilyIsInvalidate() {
    for(FamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsInvalidate();
  }

  void _notifyFamilyIsIncreased() {
    for(FamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsIncreased();
  }

  void _notifyGroupHasChanged() {
    throw FatalErrorException(A_FUNCINFO, "Group changes while registered in AnyItem::Family");
  }

private:
  
  //! Conteneur des groupes
  Private::GroupIndexMapping m_groups;
  
  //! Indentifiant maximal (équivalent à la taille de la famille)
  Integer m_max_local_id;
  
  //! Pour que les objets construits sur la famille ne puissent pas la modifier
  mutable FamilyObservers m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Famille AnyItem (pattern flyweight)
 * Aggrégation de groupes pour décrire des variables / variables partielles
 * Recopie par référence
 */
class Family
{ 
public:
  
  Family() : m_internal(new FamilyInternal) {} 
  Family(const Family& f) : m_internal(f.m_internal) {} 
  ~Family() {}
  
public:
  
  //! Comparaisons
  bool operator==(const Family& f) const { return m_internal == f.m_internal; }
  bool operator!=(const Family& f) const { return !operator==(f); }
  
  Family& operator=(const Family& f) 
  {
    m_internal = f.m_internal;
    return *this;
  }

  //! Ajout d'un groupe dans la famille
  Family& operator<<(GroupBuilder builder)
  {
    *m_internal << builder;
    return *this;
  }

  //! retroune vrai si la famille contient le groupe
  inline bool contains(const ItemGroup& group) const 
  {
    return m_internal->contains(group);
  }
  
  //! retourne vrai si le groupe est associé à une variable partielle
  inline bool isPartial(const ItemGroup& group) const 
  {
    return m_internal->isPartial(group);
  }
  
  //! Groupe de tous les items
  inline Group allItems() {
    return m_internal->allItems();
  }
  
  //! Position du groupe dans la famille
  inline Integer groupIndex(const ItemGroup& group) const {
    return m_internal->groupIndex(group);
  }
  
  //! Position dans la famille du premier localId de ce groupe
  inline Integer firstLocalId(const ItemGroup& group) const {
    return m_internal->firstLocalId(group);
  }

  //! Retoune l'item concret associé à ce AnyItem
  template<typename AnyItemT>
  Item item(const AnyItemT & any_item) const {
    return m_internal->item(any_item);
  }

  //! Taille de la famille, ie nombre de groupes
  inline Integer groupSize() const { 
    return m_internal->groupSize();
  }
  
  //! Nombre d'items dans cette famille
  /*! Somme de la taille de tous les groupes la composant */
  inline Integer maxLocalId() const {
    return m_internal->maxLocalId();
  }

  //! Accesseur au i-ème groupe de la famille
  ItemGroup group(Integer i) const {
    return m_internal->group(i);
  }

  //! Vide la famille
  void clear() {
    m_internal->clear();
  }

  //! Enregistre un observeur
  void registerObserver(IFamilyObserver& observer) const
  {
    m_internal->registerObserver(observer);
  }
  
  //! Supprime un observeur
  void removeObserver(IFamilyObserver& observer) const
  {
    m_internal->removeObserver(observer);
  }

  FamilyInternal * internal() const {
    return m_internal.get();
  }

private:
  
  //! Famille interne
  SharedPtrT<FamilyInternal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMFAMILY_H */
