// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVector.h                                                (C) 2000-2024 */
/*                                                                           */
/* Vector of entities of the same type.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMVECTOR_H
#define ARCANE_CORE_ITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemVectorView.h"
#include "arcane/core/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Entity vector.
 *
 * The ItemVector class uses a reference semantics.
 *
 * \note This class is not thread-safe and should not be used by
 * different threads simultaneously.
 *
 * \warning A vector must necessarily be associated with an entity family
 * (ItemFamily*) before being used. This can be done either via
 * calling setFamily(), or via a constructor that takes a family
 * as an argument.
 *
 * \a ItemVector is the generic class. It is possible to have a
 * specialized version by entity type via ItemVectorT.
 *
 * The operation of the entity vector is similar to that of
 * the ItemGroup entity group, with the following differences:
 * - the vector is local to the subdomain
 * - the vector is invalidated if the associated family (IItemFamily) changes
 * (after calling IItemFamily::endUpdate() if compaction or sorting is
 * active)
 * - the vector can contain the same entity multiple times.
 *
 * A vector is useful for building a temporary list
 * of entities. It inherits functionalities similar to the Array class
 * and it is therefore possible, for example, to add elements one by one,
 * either via a localId(), or via an entity.
 */
class ARCANE_CORE_EXPORT ItemVector
{
 public:

  using ItemType = Item;

 public:

  //! Creates an empty vector associated with the family \a family.
  explicit ItemVector(IItemFamily* afamily);

  //! Creates a vector associated with the family \a family and containing the entities \a local_ids.
  ItemVector(IItemFamily* afamily, Int32ConstArrayView local_ids);

  //! Creates a vector for \a size elements associated with the family \a family.
  ItemVector(IItemFamily* afamily, Integer asize);

  //! Creates a null vector. You must then call setFamily() to use it
  ItemVector();

 public:

  //! Cast operator to ItemVectorView
  operator ItemVectorView() const { return view(); }

 public:

  /*!
   * \brief Sets the associated family
   *
   * The vector is cleared of its elements
   */
  void setFamily(IItemFamily* afamily);

  //! Adds an entity with local ID \a local_id to the end of the vector
  void add(Int32 local_id) { m_local_ids.add(local_id); }

  //! Adds a list of entity local IDs \a local_ids to the end of the vector
  void add(ConstArrayView<Int32> local_ids) { m_local_ids.addRange(local_ids); }

  //! Adds an entity with local ID \a local_id to the end of the vector
  void addItem(ItemLocalId local_id) { m_local_ids.add(local_id); }

  //! Adds an entity to the end of the vector
  void addItem(Item item) { m_local_ids.add(item.localId()); }

  //! Number of elements in the vector
  Int32 size() const { return m_local_ids.size(); }

  //! Reserves memory for \a capacity entities
  void reserve(Integer capacity) { m_local_ids.reserve(capacity); }

  //! Removes all entities from the vector.
  void clear() { m_local_ids.clear(); }

  //! View of the vector
  ItemVectorView view() const { return ItemVectorView(m_shared_info, m_local_ids, 0); }

  //! View of the local IDs
  ArrayView<Int32> viewAsArray() { return m_local_ids.view(); }

  //! Constant view of the local IDs
  ConstArrayView<Int32> viewAsArray() const { return m_local_ids.constView(); }

  //! Removes the entity at index \a index
  void removeAt(Int32 index) { m_local_ids.remove(index); }

  /*!
   * \brief Sets the number of elements in the array.
   *
   * If the new size is greater than the old size, the
   * added elements are undefined.
   */
  void resize(Integer new_size) { m_local_ids.resize(new_size); }

  //! Clones this vector
  ItemVector clone() { return ItemVector(m_family, m_local_ids.constView()); }

  //! Entity at position \a index of the vector
  Item operator[](Int32 index) const { return Item(m_local_ids[index], m_shared_info); }

  //! Family associated with the vector
  IItemFamily* family() const { return m_family; }

  //! Enumerator
  ItemEnumerator enumerator() const { return ItemEnumerator(m_shared_info, m_local_ids); }

 protected:

  SharedArray<Int32> m_local_ids;
  IItemFamily* m_family = nullptr;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Typed entity vector.
 *
 * For more information, see ItemVector.
 */
template <typename VectorItemType>
class ItemVectorT
: public ItemVector
{
 public:

  using ItemType = VectorItemType;

 public:

  //! Empty constructor
  ItemVectorT() = default;

  //! Empty constructor with family
  explicit ItemVectorT(IItemFamily* afamily)
  : ItemVector(afamily)
  {}

  //! Creates a vector associated with the family \a afamily and containing the entities \a local_ids.
  ItemVectorT(IItemFamily* afamily, ConstArrayView<Int32> local_ids)
  : ItemVector(afamily, local_ids)
  {}

  //! Copy constructor
  ItemVectorT(const ItemVector& rhs)
  : ItemVector(rhs)
  {}

  //! Constructor for \a asize elements for the family \a afamily
  ItemVectorT(IItemFamily* afamily, Integer asize)
  : ItemVector(afamily, asize)
  {}

 public:

  //! Cast operator to ItemVectorView
  operator ItemVectorViewT<VectorItemType>() const { return view(); }

 public:

  //! Entity at position \a index of the vector
  ItemType operator[](Int32 index) const
  {
    return ItemType(m_local_ids[index], m_shared_info);
  }

  //! Adds an entity to the end of the vector
  void addItem(ItemType item) { m_local_ids.add(item.localId()); }

  //! Adds an entity to the end of the vector
  void addItem(ItemLocalIdT<ItemType> local_id) { m_local_ids.add(local_id); }

  //! View of the entire array
  ItemVectorViewT<ItemType> view() const
  {
    return ItemVectorViewT<ItemType>(m_shared_info, m_local_ids.constView(), 0);
  }

  //! Enumerator
  ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info, m_local_ids);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemVectorViewT<ItemType>::
ItemVectorViewT(const ItemVectorT<ItemType>& rhs)
: ItemVectorView(rhs.view())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
