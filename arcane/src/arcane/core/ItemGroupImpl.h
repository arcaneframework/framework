// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImpl.h                                             (C) 2000-2025 */
/*                                                                           */
/* Implementation of a mesh entity group.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMGROUPIMPL_H
#define ARCANE_ITEMGROUPIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SharedReference.h"
#include "arcane/utils/SharedPtr.h"

#include "arcane/core/GroupIndexTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro to detect design changes in ItemGroupImpl
#define ITEMGROUP_USE_OBSERVERS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemGroupObserver;
class IObservable;
class ItemGroupComputeFunctor;
class IMesh;
class ItemGroupInternal;
class ItemPairGroupImpl;
class GroupIndexTable;
class IVariableSynchronizer;
class ItemGroupImplInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Brief: Implementation of a mesh entity group.

 A group is a set of mesh entities (nodes, faces, cells, etc.)
 of the same type.

 An instance of this class should not be used directly, but
 through an instance of ItemGroup.

 An element entity can only be present once.

 The developer should not directly keep instances of this
 class but go through an ItemGroup. Since some groups are determined
 dynamically based on the content of another group (for example, the groups
 of entities specific to subdomains are calculated from the group of
 all entities of the subdomain), this ensures that group updates
 are performed correctly.

 This instance is managed by a reference counter and is destroyed
 automatically when it reaches zero.
 */
class ARCANE_CORE_EXPORT ItemGroupImpl
: public SharedReference
{
 private:

  friend class ItemGroupSubPartsByType;
  friend ItemGroup;
  class ItemSorter;

 public:

  //! Constructs a null group.
  ItemGroupImpl();

  /*! \brief Brief: Constructs a group.
   * Constructs an empty group named \a name, associated with the family \a family.
   */
  ItemGroupImpl(IItemFamily* family,const String& name);

  /*! \brief Brief: Constructs a child group of another group.
   * Constructs a group named \a name which is a child of group \a parent. The type of this
   * group is the same as that of the family it belongs to.
   */
  ItemGroupImpl(IItemFamily* family,ItemGroupImpl* parent,const String& name);

  virtual ~ItemGroupImpl(); //!< Releases resources.

 private:

  static ItemGroupImpl* shared_null;

 public:

  static ItemGroupImpl* checkSharedNull();

 public:

  virtual ISharedReference& sharedReference() { return *this; }

 public:

  //! Group name.
  const String& name() const;

  //! Full group name (with mesh + family).
  const String& fullName() const;

  //! Number of references on the group.
  virtual Integer nbRef() const { return refCount(); }

  //! Parent group (0 if none).
  ItemGroupImpl* parent() const;

  //! Returns \a true if the group is null.
  bool null() const;

  //! Returns whether the group contains only elements specific to the subdomain.
  bool isOwn() const;

  //! Sets whether the group property is local or not.
  void setOwn(bool v);

  //! Group of entities owned by the entities of this group.
  ItemGroupImpl* ownGroup();

  //! Items in the group not owned by the subdomain
  ItemGroupImpl* ghostGroup();

  // Items in the group lying on the boundary between two subdomains
  // Implemented for faces only
  ItemGroupImpl* interfaceGroup();

  //! Group of nodes of the elements of this group.
  ItemGroupImpl* nodeGroup();

  //! Group of edges of the elements of this group.
  ItemGroupImpl* edgeGroup();

  //! Group of faces of the elements of this group.
  ItemGroupImpl* faceGroup();

  //! Group of cells of the elements of this group.
  ItemGroupImpl* cellGroup();

  //! Creates a calculated subgroup
  /*! The memory management of the functor is then delegated to the group */
  ItemGroupImpl* createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor* functor);

  //! Accesses a calculated subgroup
  /*! The memory management of the functor is then delegated to the group */
  ItemGroupImpl* findSubGroup(const String& suffix);

  /*!
   *  \brief Brief: Group of internal faces of the elements of this group.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is internal if it is connected to two cells of this group.
   */
  ItemGroupImpl* innerFaceGroup();

  /*!
   *  \brief Brief: Group of external faces of the elements of this group.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * A face is external if it is connected to only one cell of this group.
   */
  ItemGroupImpl* outerFaceGroup();

  //! AMR
  /*!
   *  \brief Brief: Group of active cells of this group.
   *
   * An active cell is a leaf cell in the AMR tree.
   */
  ItemGroupImpl* activeCellGroup();

  /*!
   *  \brief Brief: Group of own active cells of this group.
   */
  ItemGroupImpl* ownActiveCellGroup();

  /*!
   *  \brief Brief: Group of active cells of this group.
   *
   * An active cell is a leaf cell in the AMR tree.
   */
  ItemGroupImpl* levelCellGroup(const Integer& level);

  /*!
   *  \brief Brief: Group of own active cells of this group.
   */
  ItemGroupImpl* ownLevelCellGroup(const Integer& level);

  /*!
   *  \brief Brief: Group of domain-specific active faces.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * An active face is internal if it is connected to two active
   * cells of this group.
   */
  ItemGroupImpl* activeFaceGroup();

  /*!
   *  \brief Brief: Group of active external faces of the elements of this group.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * An active face is external if it is connected to only one cell
   * of this group and is active.
   */
  ItemGroupImpl* ownActiveFaceGroup();

  /*!
   *  \brief Brief: Group of active internal faces of the elements of this group.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * An active face is internal if it is connected to two active cells of this group.
   */
  ItemGroupImpl* innerActiveFaceGroup();

  /*!
   * \brief Brief: Group of active external faces of the elements of this group.
   *
   * This group only exists for a cell group (itemKind()==IK_Cell).
   * An active face is external if it is connected to only one cell of this
   * group and is active.
   */
  ItemGroupImpl* outerActiveFaceGroup();

  //! AMR OFF

  //! True if the group is local to the subdomain.
  bool isLocalToSubDomain() const;

  //! Sets the boolean indicating if the group is local to the subdomain.
  void setLocalToSubDomain(bool v);

  //! Mesh to which the group belongs (0 for the null group).
  IMesh* mesh() const;

  //! Type of the group. It is the type of its elements.
  eItemKind itemKind() const;

  //! Family to which the group belongs (or 0 if none)
  IItemFamily* itemFamily() const;

  //! Number of entities in the group
  Integer size() const;

  //! True if the group is empty
  bool empty() const;

  //! Removes the entities from the group
  void clear();

  //! Parent group
  ItemGroup parentGroup();

  /*!
   * \brief Invalidates the group
   *
   * A very aggressive operation that causes the invalidation of all dependencies,
   * both of observers and constructed sub-groups.
   */
  void invalidate(bool force_recompute);

  /*!
   * \brief Adds entities with local IDs \a items_local_id.
   * \sa ItemGroup::addItems()
   */
  void addItems(Int32ConstArrayView items_local_id,bool check_if_present);

  //! Positions the group entities at \a items_local_id
  void setItems(Int32ConstArrayView items_local_id);

  //! Positions the group entities at \a items_local_id, optionally sorting them.
  void setItems(Int32ConstArrayView items_local_id,bool do_sort);

  //! Removes the entities \a items_local_id from the group
  void removeItems(Int32ConstArrayView items_local_id,bool check_if_present);
 
  //! Removes and adds the entities \a removed_local_id and \a added_local_id from the group
  void removeAddItems(Int32ConstArrayView removed_local_id,
                      Int32ConstArrayView added_local_id,
                      bool check_if_present);

  /*!
   * \brief Removes entities from the group whose isSuppressed() flag is true
   */
  void removeSuppressedItems();

  //! Checks that the group is valid.
  void checkValid();

  /*! \brief Updates the group if necessary.
   *
   A group must be updated when it becomes invalid, for example
   after calling invalidate().
   \retval true if the group was updated,
   \retval false otherwise.
   */
  bool checkNeedUpdate();

  //! List of local IDs of the entities in this group.
  Int32ConstArrayView itemsLocalId() const;

  /*!
   * \brief Starts a transaction.
   *
   * A transaction allows write access to protected groups.
   * Using this mechanism indicates to Arcane that the user 
   * is aware that they are modifying a group 'at their own risk'.
   */ 
  void beginTransaction();

  //! Ends a transaction
  void endTransaction();

  ARCANE_DEPRECATED_REASON("Y2022: Use itemInfoListView() instead")
  //! List of entities that the group relies on
  ItemInternalList itemsInternal() const;

  //! List of entities that the group relies on
  ItemInfoListView itemInfoListView() const;

  /*!
   * \internal
   * \brief Indicates to this group that it is the group of all entities
   * in the family.
   */
  void setIsAllItems();

  //! Indicates if the group is the group of all entities
  bool isAllItems() const;

  //! Changes the indices of the group entities
  void changeIds(Int32ConstArrayView old_to_new_ids);

  //! Applies the operation \a operation to the group entities.
  void applyOperation(IItemOperationByBasicType* operation);

  //! Indicates if the group structurally needs parallel synchronization
  bool needSynchronization() const;

  //! Returns the group's timestamp. This time is incremented after every modification.
  Int64 timestamp() const;

  /*!
   * \brief Attaches an observer.
   *
   * \param ref reference of the observer emitter
   * \param obs Observer
   */
  void attachObserver(const void * ref, IItemGroupObserver * obs);

  /*!
   * \brief Detaches an observer.
   *
   * \param ref reference of the observer emitter
   */
  void detachObserver(const void * ref);

  /*!
   * \brief Indicates if the content of this group is observed.
   *
   * This has the effect of enabling incremental modification mechanisms.
   * 
   *  A group may only be observed for its structure 
   *  by non-incrementally recalculated objects.
   */
  bool hasInfoObserver() const;

  //! Defines a group calculation function
  void setComputeFunctor(IFunctor* functor);

  //! Indicates if the group is calculated
  bool hasComputeFunctor() const;

  /*!
   * \brief Destroys the group. After this call, the group becomes a null group.
   *
   * \warning This method should only be called with extreme caution
   * even in Arcane's low-level code. If references remain on this group
   * the behavior is undefined.
   */
  void destroy();

  //! Table of local IDs to a position for all entities in the group
  SharedPtrT<GroupIndexTable> localIdToIndex();
 
  //! Group synchronizer
  IVariableSynchronizer* synchronizer();
  
  //! Indicates if this group has a synchronizer
  bool hasSynchronizer();

  /*!
   * \brief Checks and returns whether the group is sorted by increasing uniqueId().
   */
  bool checkIsSorted() const;

  //! \deprecated Use isContiguousLocalIds() instead
  bool isContigousLocalIds() const { return isContiguousLocalIds(); }

  //! Indicates if the group entities have contiguous localIds().
  bool isContiguousLocalIds() const;

  //! \deprecated Use checkLocalIdsAreContiguous() instead
  void checkLocalIdsAreContigous() const { return checkLocalIdsAreContiguous(); }

  /*!
   * \brief Checks if the group entities have contiguous localIds().
   *
   * If so, \a isContiguousLocalIds() will return \a true.
   */
  void checkLocalIdsAreContiguous() const;

  /*!
   * \brief Limits the maximum memory used by the group.
   *
   * If the group is a calculated group, it is invalidated and all its
   * allocated memory is released.
   *
   * If the group is a user-created group (i.e., persistent),
   * it ensures that the consumed memory is minimal. Normally, %Arcane allocates
   * a few more elements than necessary to avoid frequent reallocations.
   */
  void shrinkMemory();

  //! Number of allocated elements
  Int64 capacity() const;

  //! Internal Arcane API
  ItemGroupImplInternal* _internalApi() const;

 public:

  /*!
   * \internal
   * \brief List of local IDs of the entities in this group.
   * \warning Use with extreme caution, generally
   * only by the recalculation functor.
   */
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  Int32Array& unguardedItemsLocalId(const bool self_invalidate = true);


 public:

  //! Internal
  static void _buildSharedNull();
  //! Internal
  static void _destroySharedNull();

 private:

  //! Method for calculating sub-groups by type
  void _computeChildrenByType();
  //! Sub-group invalidation
  void _executeExtend(const Int32ConstArrayView * info);
  //! Sub-group invalidation
  void _executeReduce(const Int32ConstArrayView * info);
  //! Sub-group invalidation
  void _executeCompact(const Int32ConstArrayView * info);
  //! Sub-group invalidation
  void _executeReorder(const Int32ConstArrayView * info);
  //! Sub-group invalidation
  void _executeInvalidate();
  //! Forced update of the restructuring information flag
  void _updateNeedInfoFlag(const bool flag);
  //! Recursive forced invalidation
  /*! Does not notify observers. Must be followed by a normal invalidate() */
  void _forceInvalidate(const bool self_invalidate);

  void _checkUpdateSimdPadding();
  //! Notification from SharedReference indicating that the instance must be destroyed.
  virtual void deleteMe();

 private:

 ItemGroupInternal* m_p = nullptr; //!< Group implementation

 private:

  //! Removes the entities \a items_local_id from the group
  void _removeItems(SmallSpan<const Int32> items_local_id);
  bool _checkNeedUpdateNoPadding();
  bool _checkNeedUpdateWithPadding();
  bool _checkNeedUpdate(bool do_padding);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
