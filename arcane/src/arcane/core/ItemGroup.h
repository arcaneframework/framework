// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroup.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Mesh entity groups.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMGROUP_H
#define ARCANE_ITEMGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AutoRef.h"
#include "arcane/utils/Iterator.h"

#include "arcane/core/ItemGroupImpl.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemVectorView;
class IVariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//NOTE: the more complete documentation is in ItemGroup.cc

/*!
 * \ingroup Mesh
 * \brief Mesh entity group.
 *
 * An entity group contains a set of entities of a given family. A group is
 * created via the corresponding family using IItemFamily::createGroup(),
 * IItemFamily::findGroup().
 *
 */
class ARCANE_CORE_EXPORT ItemGroup
{
 public:
	
  //! Constructs a null group.
  ItemGroup();

  //! Constructs a group from the internal representation \a prv
  // TODO: this constructor should be explicit to avoid implicit conversions
  // but we haven't added it yet for compatibility reasons
  /*explicit*/ ItemGroup(ItemGroupImpl* prv);
  //! Constructs a reference to the group \a from.
  ItemGroup(const ItemGroup& from) : m_impl(from.m_impl) {}

  //! Assigns a reference to the group \a from to this instance.
  ItemGroup& operator=(const ItemGroup& from) = default;

  //! Iteration range type (to be removed)
  typedef ItemEnumerator const_iter;

 public:

  //! \a true means the group is the null group
  inline bool null() const
  {
    return m_impl->null();
  }

  //! Group name
  inline const String& name() const
  {
    return m_impl->name();
  }

  //! Group name
  inline const String& fullName() const
  {
    return m_impl->fullName();
  }
		
  //! Number of elements in the group
  inline Integer size() const
  {
    m_impl->_checkNeedUpdateNoPadding();
    return m_impl->size();
  }

  /*!
   * \brief Tests if the group is empty.
   *
   * A group is empty if it is null (null() returns \c true)
   * or if it has no elements (size() returns \c 0).
   * \retval true if the group is empty,
   * \retval false otherwise.
   */
  inline bool empty() const
  {
    m_impl->_checkNeedUpdateNoPadding();
    return m_impl->empty();
  }

  //! Group kind. This is the kind of its elements
  inline eItemKind itemKind() const { return m_impl->itemKind(); }

 public:

  /*!
   * \internal
   * \brief Returns the group implementation.
   * \warning This method returns a pointer to the internal representation
   * of the group and should not be used
   * outside of Arcane.
   */
  ItemGroupImpl* internal() const { return m_impl.get(); }

  //! Entity family to which this group belongs (0 for the null group)
  IItemFamily* itemFamily() const { return m_impl->itemFamily(); }

  //! Mesh to which this group belongs (0 for the null group)
  IMesh* mesh() const { return m_impl->mesh(); }

 public:

  // Items in the group owned by the subdomain
  ItemGroup own() const;

  // Items in the group not owned by the subdomain
  ItemGroup ghost() const;

  //! Returns whether the group contains only elements belonging to the subdomain
  bool isOwn() const;

  //! Sets whether the group property is local or not.
  void setOwn(bool v);

  // Items in the group lying on the boundary between two subdomains
  // Implemented for faces only
  ItemGroup interface() const;

  //! Group of nodes of the elements of this group
  NodeGroup nodeGroup() const;

  //! Group of edges of the elements of this group
  EdgeGroup edgeGroup() const;

  //! Group of faces of the elements of this group
  FaceGroup faceGroup() const;

  //! Group of cells of the elements of this group
  CellGroup cellGroup() const;

  /*!
   * \brief Group of internal faces of the elements of this group
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is internal if it is connected to two cells of this group.
   */
  FaceGroup innerFaceGroup() const;

  /*!
   * \brief Group of external faces of the elements of this group
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is external if it is connected to only one cell of this group.
   */
  FaceGroup outerFaceGroup() const;

  //! AMR
  //! Group of active cells of the elements of this group
  CellGroup activeCellGroup() const;

  //! Group of own active cells of the elements of this group
  CellGroup ownActiveCellGroup() const;

  //! Group of level l cells of the elements of this group
  CellGroup levelCellGroup(const Integer& level) const;

  //! Group of own level l cells of the elements of this group
  CellGroup ownLevelCellGroup(const Integer& level) const;

  /*!
   *  \brief Group of active faces
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   */
  FaceGroup activeFaceGroup() const;

  /*!
   * \brief Group of active faces belonging to the domain of the elements of this group
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   */
  FaceGroup ownActiveFaceGroup() const;

  /*!
   * \brief Group of internal faces of the elements of this group
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is internal if it is connected to two active cells of this group.
   */
  FaceGroup innerActiveFaceGroup() const;

  /*!
   * \brief Group of active external faces of the elements of this group
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is external if it is connected to only one cell of this group and is active.
   */
  FaceGroup outerActiveFaceGroup() const;

  //! Creates a computed subgroup
  /*! The memory management of the functor is then delegated to the group */
  ItemGroup createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor* functor) const;

  //! Access to a subgroup
  ItemGroup findSubGroup(const String& suffix) const;

  //! True if the group is local to the subdomain
  bool isLocalToSubDomain() const
  {
    return m_impl->isLocalToSubDomain();
  }

  /*! \brief Sets the boolean indicating if the group is local to the subdomain.
    
    By default upon creation, a group is common to all subdomains,
    meaning that each subdomain must possess an instance of this group,
    even if that instance is empty.
    
    A group local to the subdomain is not transferred during a rebalancing.
  */
  void setLocalToSubDomain(bool v)
  {
    m_impl->setLocalToSubDomain(v);
  }

  /*! \brief Invalidates the group.
    
    For a dynamically computed group (such as the group of elements belonging
    to the subdomain), this means it must be recalculated.
    
    If \a force_recompute is false, the group is just invalidated and will be
    recalculated the first time it is accessed. Otherwise, it is immediately
    recalculated.
  */
  void invalidate(bool force_recompute=false) { m_impl->invalidate(force_recompute); }

  //! Adds entities.
  void addItems(Int32ConstArrayView items_local_id,bool check_if_present=true);

  //! Removes entities.
  void removeItems(Int32ConstArrayView items_local_id,bool check_if_present=true);

  //! Sets the entities of the group.
  void setItems(Int32ConstArrayView items_local_id);

  /*!
   * \brief Sets the entities of the group.
   *
   * If \a do_sort is true, the entities are sorted by increasing uniqueId.
   */
  void setItems(Int32ConstArrayView items_local_id,bool do_sort);

  //! Internal check of group validity
  void checkValid();

  //! Clears the entities of the group
  void clear();

  //! Applies the operation \a operation to the entities of the group.
  void applyOperation(IItemOperationByBasicType* operation) const;

  //! View of the group entities.
  ItemVectorView view() const;

  //! Indicates if the group is that of all entities
  bool isAllItems() const;

  /*!
   * Returns the last modification time of the group.
   *
   * This time is automatically incremented after each modification.
   * It is possible to increment it manually via the call
   * to incrementTimestamp().
   */
  Int64 timestamp() const
  {
    return m_impl->timestamp();
  }
  
  /*!
   * \brief Increments the last modification time of the group.
   *
   * Normally this time is incremented automatically. However, it is
   * possible to do it manually in case of external modification of
   * the group information.
   */
  void incrementTimestamp() const;

  //! Table of local ids to a position for all entities in the group
  SharedPtrT<GroupIndexTable> localIdToIndex() const
  {
    return m_impl->localIdToIndex();
  }
 
  //! Group synchronizer
  IVariableSynchronizer* synchronizer() const;

  //! True if it is an automatically computed group.
  bool isAutoComputed() const;

  //! Indicates if the group has an active synchronizer
  bool hasSynchronizer() const;

  //! Checks and returns whether the group is sorted by increasing uniqueId().
  bool checkIsSorted() const;

  //! View of the group entities with padding for vectorization
  ItemVectorView _paddedView() const;

  /*!
   * \brief View of the group entities without padding for vectorization.
   *
   * The returned view MUST NOT be used in vectorization macros
   * such as ENUMERATE_SIMD_CELL().
   */
  ItemVectorView _unpaddedView() const;

 public:

  //! Internal Arcane API
 ItemGroupImplInternal* _internalApi() const;

 public:

  //! Enumerator over the group entities.
  ItemEnumerator enumerator() const;

 private:

  template <typename T>
  friend class SimdItemEnumeratorContainerTraits;
  //! Enumerator over the group entities for vectorization
  ItemEnumerator _simdEnumerator() const;

 protected:

  //! Internal representation of the group.
  AutoRefT<ItemGroupImpl> m_impl;

 protected:

  //! Returns the group \a impl if it is of kind \a kt, the null group otherwise
  static ItemGroupImpl* _check(ItemGroupImpl* impl,eItemKind ik)
  {
    return impl->itemKind()==ik ? impl : ItemGroupImpl::checkSharedNull();
  }

  ItemVectorView _view(bool do_padding) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compares the references of two groups.
 * \retval true if \a g1 and \a g2 refer to the same group,
 * \retval false otherwise.
 */
inline bool
operator==(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()==g2.internal();
}

/*!
 * \brief Compares two groups.
 * The order used is arbitrary and is only used for potential sorting in STL
 * containers.
 * \retval true if \a g1 is less than \a g2,
 * \retval false otherwise.
 */
inline bool
operator<(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()<g2.internal();
}

/*!
 * \brief Compares the references of two groups.
 * \retval true if \a g1 and \a g2 do not refer to the same group,
 * \retval false otherwise.
 */
inline bool
operator!=(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()!=g2.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reference to a group of a given kind.
 */
template<typename T>
class ItemGroupT
: public ItemGroup
{
 public:

  //! Type of this class
  typedef ItemGroupT<T> ThatClass;
  //! Type of the class containing the entity characteristics
  typedef ItemTraitsT<T> TraitsType;

  typedef typename TraitsType::ItemType ItemType;

  typedef const ItemType* const_iterator;
  typedef ItemType* iterator;
  typedef ItemType value_type;
  typedef const ItemType& const_reference;

 public:

  inline ItemGroupT() = default;
  inline explicit ItemGroupT(ItemGroupImpl* from)
  : ItemGroup(_check(from,TraitsType::kind())){}
  inline ItemGroupT(const ItemGroup& from)
  : ItemGroup(_check(from.internal(),TraitsType::kind())) {}
  inline ItemGroupT(const ItemGroupT<T>& from)
  : ItemGroup(from) {}
  inline const ItemGroupT<T>& operator=(const ItemGroupT<T>& from)
  { m_impl = from.internal(); return (*this); }
  inline const ItemGroupT<T>& operator=(const ItemGroup& from)
  { _assign(from); return (*this); }

 public:
  
  ThatClass own() const
  {
    return ThatClass(ItemGroup::own());
  }

  ItemEnumeratorT<T> enumerator() const
  {
    return ItemEnumeratorT<T>::fromItemEnumerator(ItemGroup::enumerator());
  }

 protected:

  void _assign(const ItemGroup& from)
  {
    m_impl = _check(from.internal(),TraitsType::kind());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
