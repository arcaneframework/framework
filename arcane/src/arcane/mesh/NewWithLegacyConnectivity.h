// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivityNewWithDependenciesInfo.h                       (C) 2000-2024 */
/*                                                                           */
/* Info for new connectivity mode (with dependencies)                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESINFO_H
#define ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// todo rename the file in NewWithLegacyConnectivity

/*---------------------------------------------------------------------------*/
/*! \brief class holding a new connectivity but also filling the legacy one
 *  Both custom and legacy connectivities of ItemConnectivitySelector are built.
 *
 */

template <class SourceFamily, class TargetFamily, class LegacyType, class CustomType = IncrementalItemConnectivity>
class ARCANE_MESH_EXPORT NewWithLegacyConnectivity
: public ItemConnectivitySelectorT<LegacyType, CustomType>
, public ReferenceCounterImpl
, public IIncrementalItemConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  friend class ConnectivityItemVector;

 public:

  NewWithLegacyConnectivity(ItemFamily* source_family, IItemFamily* target_family, const String& name)
  : ItemConnectivitySelectorT<LegacyType, CustomType>(source_family, target_family, name)
  {
    //build selector
    Base::template build<SourceFamily, TargetFamily>(); // Create adapted custom connectivity
  }

  typedef ItemConnectivitySelectorT<LegacyType, CustomType> Base;

  String name() const override { return Base::trueCustomConnectivity()->name(); }

  bool isEmpty() const
  {
    return false;
  }

  //! List of families (sourceFamily() + targetFamily())
  ConstArrayView<IItemFamily*> families() const override { return Base::trueCustomConnectivity()->families(); }

  //! Source family
  IItemFamily* sourceFamily() const override { return Base::trueCustomConnectivity()->sourceFamily(); }

  //! Target family
  IItemFamily* targetFamily() const override { return Base::trueCustomConnectivity()->targetFamily(); }

  //! Adds the entity with localId() \a target_local_id to the connectivity of \a source_item
  void addConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override { Base::addConnectedItem(source_item, target_local_id); }

  //! Removes the entity with localId() \a target_local_id from the connectivity of \a source_item
  void removeConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override { Base::removeConnectedItem(source_item, target_local_id); }

  //! Removes all entities connected to \a source_item
  void removeConnectedItems(ItemLocalId source_item) override { Base::removeConnectedItems(source_item); }

  //! Replaces the entity at index \a index of \a source_item with the entity with localId() \a target_local_id
  // Why this name difference in ItemConnectivitySelector?
  void replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_local_id) override { Base::replaceItem(source_item, index, target_local_id); }

  //! Replaces the entities of \a source_item with the entities with localId() \a target_local_ids
  // Why this name difference in ItemConnectivitySelector?
  void replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) override { Base::replaceItems(source_item, target_local_ids); }

  //! Tests the existence of a connectivity between \a source_item and the entity with localId() \a target_local_id
  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override { return Base::hasConnectedItem(source_item, target_local_id); };

  //TODO: use an event mechanism.
  //! Notifies the connectivity that the source family has been compacted.
  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override { Base::trueCustomConnectivity()->notifySourceFamilyLocalIdChanged(new_to_old_ids); }

  //TODO: use an event mechanism.
  //! Notifies the connectivity that the target family has been compacted.
  void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) override { Base::trueCustomConnectivity()->notifyTargetFamilyLocalIdChanged(old_to_new_ids); }

  //! Notifies the connectivity that an entity has been added to the source family.
  void notifySourceItemAdded(ItemLocalId item) override { Base::trueCustomConnectivity()->notifySourceItemAdded(item); }

  //! Notifies the connectivity that a read from a dump has been performed
  void notifyReadFromDump() override { Base::trueCustomConnectivity()->notifyReadFromDump(); }

  //! Number of pre-allocated entities for the connectivity of each entity
  Integer preAllocatedSize() const override { return Base::preAllocatedSize(); }

  //! Sets the number of entities to pre-allocate for the connectivity of each entity
  void setPreAllocatedSize(Integer value) override { Base::setPreAllocatedSize(value); }

  //! Dumps statistics on usage and memory used to the stream \a out
  void dumpStats(std::ostream& out) const override { Base::trueCustomConnectivity()->dumpStats(out); }

  //! Number of entities connected to the source entity with local number \a lid
  Integer nbConnectedItem(ItemLocalId lid) const override { return Base::trueCustomConnectivity()->nbConnectedItem(lid); }

  //! localId() of the \a index-th entity connected to the source entity with local number \a lid
  Int32 connectedItemLocalId(ItemLocalId lid, Integer index) const override { return Base::trueCustomConnectivity()->connectedItemLocalId(lid, index); }

  //! Maximum number of entities connected to a source entity.
  Int32 maxNbConnectedItem() const override { return Base::trueCustomConnectivity()->maxNbConnectedItem(); }

  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override
  {
    return Arccore::makeRef<IIncrementalItemSourceConnectivity>(this);
  }
  Ref<IIncrementalItemTargetConnectivity> toTargetReference() override
  {
    return Arccore::makeRef<IIncrementalItemTargetConnectivity>(this);
  }
  IIncrementalItemConnectivityInternal* _internalApi() override
  {
    return Base::trueCustomConnectivity()->_internalApi();
  }

 protected:

  //! Implements the initialization of \a civ for this connectivity.
  void _initializeStorage(ConnectivityItemVector* civ) override { Base::trueCustomConnectivity()->_initializeStorage(civ); };

  //! Fills \a con_items with the entities connected to \a item.
  ItemVectorView _connectedItems(ItemLocalId item, ConnectivityItemVector& con_items) const override { return Base::trueCustomConnectivity()->_connectedItems(item, con_items); }
};

template <class SourceFamily, class TargetFamily>
class ARCANE_MESH_EXPORT NewWithLegacyConnectivityType
{
 public:

  typedef NewWithLegacyConnectivity<SourceFamily, TargetFamily, typename LegacyConnectivityTraitsT<TargetFamily>::type> type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
