// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivity.h                              (C) 2000-2024 */
/*                                                                           */
/* Incremental entity connectivity interface.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_CORE_IINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the source of an incremental connectivity
 */
class ARCANE_CORE_EXPORT IIncrementalItemSourceConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IIncrementalItemSourceConnectivity() = default;

 public:

  //! Source family
  virtual IItemFamily* sourceFamily() const = 0;

  //TODO: use an event-based mechanism.
  //! Notifies the connectivity that the source family has been compacted.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) = 0;

  //! Notifies the connectivity that an entity has been added to the source family.
  virtual void notifySourceItemAdded(ItemLocalId item) = 0;

  /*!
   * \brief Reserves memory for \a n source entities.
   *
   * Calling this method is optional but prevents multiple
   * reallocations during successive calls to notifySourceItemAdded().
   *
   * If \a pre_alloc_connectivity is true, it also pre-allocates the list of
   * connectivities based on the value of preAllocatedSize(). For example,
   * if preAllocatedSize() is 4 and \a n is 10000, we will pre-allocate
   * for 40000 connectivities. To avoid unnecessary memory overconsumption,
   * connectivities should only be pre-allocated if we are sure we will use them.
   */
  virtual void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity);

  //! Notifies the connectivity that a read has been performed from a dump
  virtual void notifyReadFromDump() = 0;

  //! Returns a reference to the instance
  virtual Ref<IIncrementalItemSourceConnectivity> toSourceReference() = 0;

 private:

  // Interfaces reserved to Arcane

  //! Notifies the connectivity that the entities \a items have been added to the source family
  virtual void _internalNotifySourceItemsAdded(Int32ConstArrayView items);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the target of an incremental connectivity
 */
class ARCANE_CORE_EXPORT IIncrementalItemTargetConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IIncrementalItemTargetConnectivity() = default;

 public:

  //TODO: use an event-based mechanism.
  //! Notifies the connectivity that the target family has been compacted.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) = 0;

  //! Returns a reference to the instance
  virtual Ref<IIncrementalItemTargetConnectivity> toTargetReference() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for managing an incremental connectivity.
 *
 * A connectivity links two families: a source (sourceFamily()) and
 * a target (targetFamily()).
 */
class ARCANE_CORE_EXPORT IIncrementalItemConnectivity
: public IItemConnectivityAccessor
, public IIncrementalItemSourceConnectivity
, public IIncrementalItemTargetConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  //TODO make 'protected' once everyone uses the reference counter
  ~IIncrementalItemConnectivity() = default;

 public:

  //! Name of the connectivity
  virtual String name() const = 0;

  //! List of families (sourceFamily() + targetFamily())
  virtual ConstArrayView<IItemFamily*> families() const = 0;

  //! Target family
  virtual IItemFamily* targetFamily() const = 0;

  //! Adds the entity with localId() \a target_local_id to the connectivity of \a source_item
  virtual void addConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) = 0;

  /*!
   * \brief Allocates and positions entities connected to \a source_item.
   *
   * If there were already entities connected to \a source_item, they are removed.
   * \a target_local_ids contains the list of local IDs of entities to add.
   * This method is equivalent to calling the following code but allows for memory
   * management optimizations:
   * \code
   * IIncrementalItemConnectivity* c = ...;
   * c->removeConnectedItems(source_item);
   * for( Int32 x : target_local_ids )
   *   c->addConnectedItem(source_item,ItemLocalId{x});
   * \endcode
   */
  virtual void setConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids);

  //! Removes the entity with localId() \a target_local_id from the connectivity of \a source_item
  virtual void removeConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) = 0;

  //! Removes all entities connected to \a source_item
  virtual void removeConnectedItems(ItemLocalId source_item) = 0;

  //! Replaces the entity at index \a index of \a source_item with the entity with localId() \a target_local_id
  virtual void replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_local_id) = 0;

  //! Replaces the entities of \a source_item with the entities with localId() \a target_local_ids
  virtual void replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) = 0;

  //! Tests the existence of a connectivity between \a source_item and the entity with localId() \a target_local_id
  virtual bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const = 0;

  /*!
   * \brief Maximum number of entities connected to a source entity.
   *
   * This value may be greater than the current maximum number of connected entities
   * if removeConnectedItem() and removeConnectedItems() have been called.
   */
  virtual Int32 maxNbConnectedItem() const = 0;

  //! Number of entities pre-allocated for the connectivity of each entity
  virtual Integer preAllocatedSize() const = 0;

  //! Sets the number of entities to pre-allocate for the connectivity of each entity
  virtual void setPreAllocatedSize(Integer value) = 0;

  //! Dumps statistics on usage and memory used to the stream \a out
  virtual void dumpStats(std::ostream& out) const = 0;

  //! Internal Arcane API
  virtual IIncrementalItemConnectivityInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
