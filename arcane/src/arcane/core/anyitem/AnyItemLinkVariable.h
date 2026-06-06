// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemLinkVariable.h                                       (C) 2000-2025 */
/*                                                                           */
/* Link variable for items of arbitrary types.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMLINKVARIABLE_H
#define ARCANE_CORE_ANYITEM_ANYITEMLINKVARIABLE_H
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
 * \brief Link variable for items of arbitrary types.
 * 
 * For example:
 *
 * AnyItem::Variable<Real> variable(family);               // Filled
 * AnyItem::LinkVariable<Real> link_variable(link_family); // Filled
 *
 * Real value = 0;
 * ENUMERATE_ANY_ITEM_LINK(ilink, link_family) {
 *   if(ilink.index() < 10) {
 *      info() << "back item = [uid=" << family.concreteItem(ilink.back()).uniqueId() 
 *             << ",lid=" << family.concreteItem(ilink.back()).localId() << ",kind="
 *             << family.concreteItem(ilink.back()).kind() << "]";
 *      info() << "front item = [uid=" << family.concreteItem(ilink.front()).uniqueId() 
 *             << ",lid=" << family.concreteItem(ilink.front()).localId() << ",kind="
 *             << family.concreteItem(ilink.front()).kind() << "]";
 *    }
 *   value += link_variable[ilink] + variable[ilink.back()] + variable[ilink.front()];
 * }
 *
 */
template<typename DataType>
class LinkVariable 
  : public ILinkFamilyObserver
{
public:

  LinkVariable(const LinkFamily& family)
    : m_family(family) 
    , m_values(m_family.capacity()) 
  {
    m_family.registerObserver(*this);
  }

  LinkVariable(const LinkVariable& v)
    : m_family(v.m_family) 
    , m_values(v.m_values) 
  {
    m_family.registerObserver(*this);
  }
  
  ~LinkVariable() 
  {
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this); });
  }

  //! Accessor
  inline DataType& operator[](const LinkFamily::LinkIndex& item) {
    return m_values[item.index()];
  }
  
  //! Accessor
  inline DataType operator[](const LinkFamily::LinkIndex& item) const {
    return m_values[item.index()];
  }
 
  //! Action if the family is invalidated: we resize
  inline void notifyFamilyIsInvalidate() {
    // If the family changes, we resize
    m_values.resize(m_family.capacity());
  }
  
  //! Action if the family is reserved: we resize
  inline void notifyFamilyIsReserved() {
    // If the family is reserved, we simply resize
    m_values.resize(m_family.capacity());
  }
  
private:
  
  //! Link family
  const LinkFamily m_family;
  
  //! Values
  Arcane::UniqueArray<DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
    
#endif /* ARCANE_ANYITEM_ANYITEMLINKVARIABLEARRAY_H */
