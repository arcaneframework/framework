// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalVectorView.h                                    (C) 2000-2017 */
/*                                                                           */
/* Vue sur un vecteur (tableau indirect) d'entités.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINTERNALVECTORVIEW_H
#define ARCANE_ITEMINTERNALVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/ItemTypes.h"
#include <iterator>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterateur d'un ItemInternalVectorView.
 */
class ItemInternalVectorViewConstIterator
{
  friend class ItemInternalVectorView;
  typedef ItemInternal* ItemInternalPtr;
  ItemInternalVectorViewConstIterator(const ItemInternalPtr* items,
                                      const Int32* ARCANE_RESTRICT local_ids,
                                      Integer index)
  : m_items(items), m_local_ids(local_ids), m_index(index){}
 public:
  // Pas directement utilisé mais est nécessaire pour ICC 17.0
  ItemInternalVectorViewConstIterator()
  : m_items(nullptr), m_local_ids(nullptr), m_index(0){}
 public:
  typedef ItemInternalVectorViewConstIterator ThatClass;
 public:
  typedef std::random_access_iterator_tag iterator_category;
  //! Type indexant le tableau
  typedef const ItemInternalPtr* pointer;
  //! Type indexant le tableau
  typedef const ItemInternalPtr& reference;
  //! Type indexant le tableau
  typedef ItemInternalPtr value_type;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef Integer difference_type;
 public:
  value_type operator*() const { return m_items[ m_local_ids[m_index] ]; }
  value_type operator->() const { return m_items[ m_local_ids[m_index] ]; }
  ThatClass& operator++() { ++m_index; return (*this); }
  ThatClass& operator--() { --m_index; return (*this); }
  void operator+=(difference_type v) { m_index += v; }
  void operator-=(difference_type v) { m_index -= v; }
  friend Integer operator-(const ThatClass& a,const ThatClass& b)
  {
    return a.m_index - b.m_index;
  }
  friend ThatClass operator-(const ThatClass& a,difference_type v)
  {
    Integer index = a.m_index - v;
    return ThatClass(a.m_items,a.m_local_ids,index);
  }
  friend ThatClass operator+(const ThatClass& a,difference_type v)
  {
    Integer index = a.m_index + v;
    return ThatClass(a.m_items,a.m_local_ids,index);
  }
  friend bool operator<(const ThatClass& lhs,const ThatClass& rhs)
  {
    return lhs.m_index<=rhs.m_index;
  }
  friend bool operator==(const ThatClass& lhs,const ThatClass& rhs)
  {
    // TODO: regarder si cela ne pose pas de problemes de performance
    // de faire ces trois comparaions. Si c'est le cas on peut
    // faire uniquement la dernière.
    return lhs.m_items==rhs.m_items && lhs.m_local_ids==rhs.m_local_ids && lhs.m_index==rhs.m_index;
  }
  friend bool operator!=(const ThatClass& lhs,const ThatClass& rhs)
  {
    return !(lhs==rhs);
  }
 private:
  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vue sur un tableau indexé d'entités.
 * \see ItemVectorView
 */
class ItemInternalVectorView
{
 public:
  typedef ItemInternalVectorViewConstIterator const_iterator;
 public:

  ItemInternalVectorView() = default;

  ItemInternalVectorView(ItemInternalArrayView aitems,Int32ConstArrayView local_ids)
  : m_items(aitems), m_local_ids(local_ids) {}

  ItemInternalVectorView(ItemInternalArrayView aitems,const Int32* local_ids,Integer count)
  : m_items(aitems), m_local_ids(count,local_ids) {}

 public:

  //! Accède au \a i-ème élément du vecteur
  inline ItemInternal* operator[](Integer index) const
  {
    return m_items[ m_local_ids[index] ];
  }

  //! Nombre d'éléments du vecteur
  inline Integer size() const { return m_local_ids.size(); }

  //! Tableau des entités
  inline ItemInternalArrayView items() const { return m_items; }

  //! Tableau des numéros locaux des entités
  inline Int32ConstArrayView localIds() const { return m_local_ids; }

 public:

  inline const_iterator begin() const
  {
    return const_iterator(m_items.unguardedBasePointer(),m_local_ids.unguardedBasePointer(),0);
  }
  inline const_iterator end() const
  {
    return const_iterator(m_items.unguardedBasePointer(),m_local_ids.unguardedBasePointer(),this->size());
  }

 protected:

  ItemInternalArrayView m_items;
  Int32ConstArrayView m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
