// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemArray.h                                              (C) 2000-2012 */
/*                                                                           */
/* Tableau d'items de types quelconques.                                     */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ANYITEM_ANYITEMARRAY_H 
#define ARCANE_ANYITEM_ANYITEMARRAY_H

#include "arcane/core/IItemFamily.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Tableau d'items de types quelconques.
 * 
 * Similaire aux variables mais sans les définir
 * 
 * Par exemle :
 *
 * AnyItem::UniqueArray<Real> array(family.allItems());
 * array.fill(0.);

 * ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
 *   array[iitem] += variable[iitem];
 * }
 *
 * \TODO : on pourrait améliorer l'implémentation en utilisant le localId dans la AnyItem::Family avec un tableau unique alloué à maxLocalId
 */
template<typename DataType>
class Array
{
public:
  
  Array(const Group& group)
  {
    for(Group::Enumerator e = group.enumerator(); e.hasNext(); ++e) {
      if(e.groupIndex() >= m_values.size())
        m_values.resize(e.groupIndex()+1);
    }
    for(Group::Enumerator e = group.enumerator(); e.hasNext(); ++e) {
      m_values[e.groupIndex()].resize(e.group().itemFamily()->maxLocalId());
    }
  }
  
  //! Remplissage du tableau 
  void fill(const DataType& data) 
  {
    for(Integer i = 0; i < m_values.size(); ++i) {
      m_values[i].fill(data);
    }
  }
  
  //! Accesseur
  template<typename T>
  inline DataType& operator[](const T& item) { 
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  // Acceseur
  template<typename T>
  inline typename Arcane::UniqueArray<DataType>::ConstReferenceType operator[](const T& item) const { 
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
private:
  
  //! Conteneur des variables génériques
  Arcane::UniqueArray< Arcane::UniqueArray<DataType> > m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMARRAY_H */
