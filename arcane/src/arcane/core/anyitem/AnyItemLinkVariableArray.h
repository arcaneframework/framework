// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemLinkVariableArray.h                                  (C) 2000-2025 */
/*                                                                           */
/* Variable 2D de liens d'items de types quelconques.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMLINKVARIABLEARRAY_H
#define ARCANE_CORE_ANYITEM_ANYITEMLINKVARIABLEARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Variable de liens d'items de types quelconques.
 * 
 * Par exemple :
 *
 * AnyItem::Variable<Real> variable(family);                    // Remplie
 * AnyItem::LinkVariableArray<Real> link_variable(link_family); // Remplie de taille 3
 *
 * Real value = 0;
 * ENUMERATE_ANY_ITEM_LINK(ilink, link_family) {
 *   if(ilink.index() < 10) {
 *     info() << "back item = [uid=" << family.concreteItem(ilink.back()).uniqueId()
 *            << ",lid=" << family.concreteItem(ilink.back()).localId() << ",kind="
 *            << family.concreteItem(ilink.back()).kind() << "]";
 *     info() << "front item = [uid=" << family.concreteItem(ilink.front()).uniqueId()
 *            << ",lid=" << family.concreteItem(ilink.front()).localId() << ",kind="
 *            << family.concreteItem(ilink.front()).kind() << "]";
 *   }
 *
 *   for(Integer i = 0; i < 3; ++i)
 *     value += link_variable[ilink][i] + variable[ilink.back()][i] + variable[ilink.front()];
 * }
 *
 */
template<typename DataType>
class LinkVariableArray
  : public ILinkFamilyObserver
{
public:

  LinkVariableArray(const LinkFamily& family)
    : m_size(0)
    , m_family(family)
    , m_values(m_family.capacity(),m_size)
  {
    m_family.registerObserver(*this);
  }

  LinkVariableArray(const LinkVariableArray& v)
    : m_size(v.m_size)
    , m_family(v.m_family)
    , m_values(v.m_values) 
  {
    m_family.registerObserver(*this);
  }
  
  ~LinkVariableArray()
  {
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this); });
  }

  //! Accesseur
  inline ArrayView<DataType> operator[](const LinkFamily::LinkIndex& item) {
    return m_values[item.index()];
  }
  
  //! Accesseurmake
  inline ConstArrayView<DataType> operator[](const LinkFamily::LinkIndex& item) const {
    return m_values[item.index()];
  }
 
  //! Action si la famille est invalidée : on retaille
  inline void notifyFamilyIsInvalidate() {
    // Si la famille change, on retaille
    m_values.resize(m_family.capacity(),m_size);
  }
  
  //! Action si la famille est reservée : on retaille
  inline void notifyFamilyIsReserved() {
    // Si la famille est reservée, on retaille simplement
    m_values.resize(m_family.capacity(),m_size);
  }
  
  //! Redimensionnement de la deuxième dimension du tableau
  inline void resize(Integer size)
  {
    m_size = size;
    m_values.resize(m_family.capacity(),m_size);
  }

  //! Retourne la taille du tableau
  inline Integer size() const { return m_size; }

private:
  
  //! Taille de la 2ème dimension du tableau
  Integer m_size;

  //! Famille de liens
  const LinkFamily m_family;
  
  //! Valeurs
  Arcane::UniqueArray2<DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
    
#endif /* ARCANE_ANYITEM_ANYITEMLINKVARIABLEARRAY_H */
