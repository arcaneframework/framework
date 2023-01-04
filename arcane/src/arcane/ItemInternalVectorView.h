// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalVectorView.h                                    (C) 2000-2023 */
/*                                                                           */
/* Vue sur un vecteur (tableau indirect) d'entités.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINTERNALVECTORVIEW_H
#define ARCANE_ITEMINTERNALVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/ItemTypes.h"
#include "arcane/ItemSharedInfo.h"

#include <iterator>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Iterateur d'un ItemInternalVectorView.
 * \deprecated Utiliser un itérateur à partir d'un ItemVectorView.
 */
class ItemInternalVectorViewConstIterator
{
  friend class ItemInternalVectorView;
  typedef ItemInternal* ItemInternalPtr;
 private:
  ItemInternalVectorViewConstIterator(const ItemInternalPtr* items,
                                      const Int32* ARCANE_RESTRICT local_ids,
                                      Integer index)
  : m_items(items), m_local_ids(local_ids), m_index(index){}
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
 * \brief Vue interne sur un tableau d'entités.
 *
 * Celle classe n'est utile que pour construire des listes d'entités utilisées
 * en interne de %Arcane. La version utilisateur de cette classe est
 * ItemVectorView.
 *
 * \see ItemVectorView
 */
class ARCANE_CORE_EXPORT ItemInternalVectorView
{
 public:

  friend class ItemVectorView;
  friend class ItemInternalConnectivityList;
  friend class impl::ItemBase;
  friend class ItemEnumeratorBase;
  friend class SimdItemEnumeratorBase;
  template<typename T> friend class ItemEnumeratorBaseT;
  using const_iterator = ItemInternalVectorViewConstIterator;

 public:

  ItemInternalVectorView() = default;

 private:

  ItemInternalVectorView(ItemSharedInfo* si,Int32ConstArrayView local_ids)
  : m_local_ids(local_ids), m_shared_info(si)
  {
    ARCANE_ASSERT(_isValid(),("Bad ItemInternalVectorView"));
  }

  ItemInternalVectorView(ItemSharedInfo* si,const Int32* local_ids,Integer count)
  : m_local_ids(count,local_ids), m_shared_info(si)
  {
    ARCANE_ASSERT(_isValid(),("Bad ItemInternalVectorView"));
  }

 public:

  /*!
   * \brief Accède au \a i-ème élément du vecteur.
   *
   * Cette méthode est obsolète. Il faut construire un 'ItemVectorView' à la place
   * et utiliser l'opérateur 'operator[]' associé.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use ItemVectorView::operator[] instead")
  ItemInternal* operator[](Integer index) const { return m_shared_info->m_items_internal[ m_local_ids[index] ]; }

  //! Nombre d'éléments du vecteur
  Integer size() const { return m_local_ids.size(); }

  //! Tableau des entités
  ItemInternalArrayView items() const { return m_shared_info->m_items_internal; }

  //! Tableau des numéros locaux des entités
  Int32ConstArrayView localIds() const { return m_local_ids; }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use ItemVectorView to iterate")
  const_iterator begin() const
  {
    return const_iterator(items().data(),m_local_ids.data(),0);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Use ItemVectorView to iterate")
  const_iterator end() const
  {
    return const_iterator(items().data(),m_local_ids.data(),this->size());
  }

 protected:

  Int32ConstArrayView m_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 private:

  bool _isValid();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
