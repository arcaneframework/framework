// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemArray2.h                                             (C) 2000-2025 */
/*                                                                           */
/* 2D array of items of arbitrary types.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMARRAY2_H
#define ARCANE_CORE_ANYITEM_ANYITEMARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"

#include "arcane/core/IItemFamily.h"
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
 * \brief 2D array of items of arbitrary types.
 * 
 * Similar to 2D variables but without defining them
 * 
 * For example:
 *
 * AnyItem::UniqueArray2<Real> array(family.allItems());
 * array.resize(3);
 * array.fill(0.);
 *
 * ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
 *   for(Integer i = 0; i < 3; ++i)
 *     array[iitem][i] += variable[iitem];
 * }
 *
 * \TODO: We could improve the implementation by using localId in
 * AnyItem::Family with a unique array allocated to maxLocalId
 */
template<typename DataType>
class Array2
{
public:
  
  Array2(const Group& group)
  : m_size(0)
  {
    for(Group::Enumerator e = group.enumerator(); e.hasNext(); ++e) {
      if(e.groupIndex() >= m_values.size())
        m_values.resize(e.groupIndex()+1);
    }
    for(Group::Enumerator e = group.enumerator(); e.hasNext(); ++e) {
      m_values[e.groupIndex()].resize(e.group().itemFamily()->maxLocalId(),m_size);
    }
  }
  
  //! Resizing the second dimension of the array
  inline void resize(Integer size)
  {
    m_size = size;
    for(Integer i = 0; i < m_values.size(); ++i) {
      m_values[i].resize(m_values[i].dim1Size(),m_size);
    }
  }

  //! Filling the array 
  void fill(const DataType& data) 
  {
    for(Integer i = 0; i < m_values.size(); ++i) {
      m_values[i].fill(data);
    }
  }
  
  //! Accessor
  template<typename T>
  inline ArrayView<DataType> operator[](const T& item) {
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  // Accessor
  template<typename T>
  inline ConstArrayView<DataType> operator[](const T& item) const {
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  //! Returns the size of the array
  inline Integer size() const { return m_size; }

private:
  
  //! Size of the array's second dimension
  Integer m_size;

  //! Container for generic variables
  Arcane::UniqueArray< Arcane::UniqueArray2<DataType> > m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMARRAY2_H */
