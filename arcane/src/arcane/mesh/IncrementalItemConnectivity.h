// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalItemConnectivity.h                               (C) 2000-2024 */
/*                                                                           */
/* Incremental connectivity of entities.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INCREMENTALITEMCONNECTIVITY_H
#define ARCANE_MESH_INCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemConnectivityMemoryInfo;
}

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IncrementalItemConnectivityContainer;
class IndexedItemConnectivityAccessor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Abstract class for managing connectivities.
 *
 * This class manages common information for all types of
 * connectivity such as its name, source and target families, ...
 */
class ARCANE_MESH_EXPORT AbstractIncrementalItemConnectivity
: public TraceAccessor
, public ReferenceCounterImpl
, public IIncrementalItemConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  AbstractIncrementalItemConnectivity(IItemFamily* source_family,
                                      IItemFamily* target_family,
                                      const String& connectivity_name);

 public:

  String name() const final { return m_name; }

 public:

  ConstArrayView<IItemFamily*> families() const override { return m_families.constView(); }
  IItemFamily* sourceFamily() const override { return m_source_family; }
  IItemFamily* targetFamily() const override { return m_target_family; }

  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override;
  Ref<IIncrementalItemTargetConnectivity> toTargetReference() override;

 protected:

  ConstArrayView<IItemFamily*> _families() const { return m_families.constView(); }
  IItemFamily* _sourceFamily() const { return m_source_family; }
  IItemFamily* _targetFamily() const { return m_target_family; }

 private:

  IItemFamily* m_source_family;
  IItemFamily* m_target_family;
  SharedArray<IItemFamily*> m_families;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Base class for incremental item->item[] connectivities.
 */
class ARCANE_MESH_EXPORT IncrementalItemConnectivityBase
: public AbstractIncrementalItemConnectivity
{
  class InternalApi;

 public:

  template <class SourceFamily, class TargetFamily, class LegacyType, class CustomType>
  friend class NewWithLegacyConnectivity;

 public:

  IncrementalItemConnectivityBase(IItemFamily* source_family, IItemFamily* target_family,
                                  const String& aname);
  ~IncrementalItemConnectivityBase() override;

 public:

  // Copying is forbidden because of \a m_p
  IncrementalItemConnectivityBase(const IncrementalItemConnectivityBase&) = delete;
  IncrementalItemConnectivityBase(IncrementalItemConnectivityBase&&) = delete;
  IncrementalItemConnectivityBase& operator=(const IncrementalItemConnectivityBase&) = delete;
  IncrementalItemConnectivityBase& operator=(IncrementalItemConnectivityBase&&) = delete;

 public:

  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) override;
  Integer nbConnectedItem(ItemLocalId lid) const final
  {
    return m_connectivity_nb_item[lid];
  }
  Int32 connectedItemLocalId(ItemLocalId lid, Integer index) const final
  {
    return m_connectivity_list[m_connectivity_index[lid] + index];
  }

  IndexedItemConnectivityViewBase connectivityView() const;
  IndexedItemConnectivityAccessor connectivityAccessor() const;
  ItemConnectivityContainerView connectivityContainerView() const;

  Int32 maxNbConnectedItem() const override;

  void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity) override;
  IIncrementalItemConnectivityInternal* _internalApi() override;

 public:

  Int32ConstArrayView _connectedItemsLocalId(ItemLocalId lid) const
  {
    Int32 nb = m_connectivity_nb_item[lid];
    Int32 index = m_connectivity_index[lid];
    return { nb, &m_connectivity_list[index] };
  }

  // TODO: see if we should keep this method. Use as little as possible.
  Int32ArrayView _connectedItemsLocalId(ItemLocalId lid)
  {
    Int32 nb = m_connectivity_nb_item[lid];
    Int32 index = m_connectivity_index[lid];
    return { nb, &m_connectivity_list[index] };
  }

 public:

  Int32ArrayView connectivityIndex() { return m_connectivity_index; }
  Int32ArrayView connectivityList() { return m_connectivity_list; }

  void setItemConnectivityList(ItemInternalConnectivityList* ilist, Int32 index);
  void dumpInfos();

 protected:

  void _initializeStorage(ConnectivityItemVector* civ) override
  {
    ARCANE_UNUSED(civ);
  }
  ItemVectorView _connectedItems(ItemLocalId item, ConnectivityItemVector& con_items) const final;

 protected:

  bool m_is_empty = true;
  Int32ArrayView m_connectivity_nb_item;
  Int32ArrayView m_connectivity_index;
  Int32ArrayView m_connectivity_list;
  IncrementalItemConnectivityContainer* m_p = nullptr;
  ItemInternalConnectivityList* m_item_connectivity_list = nullptr;
  Integer m_item_connectivity_index = -1;
  std::unique_ptr<InternalApi> m_internal_api;

 protected:

  void _notifyConnectivityListChanged();
  void _notifyConnectivityIndexChanged();
  void _notifyConnectivityNbItemChanged();
  void _notifyConnectivityNbItemChangedFromObservable();
  void _computeMaxNbConnectedItem();
  void _setNewMaxNbConnectedItems(Int32 new_max);
  void _setMaxNbConnectedItemsInConnectivityList();

 private:

  void _shrinkMemory();
  void _addMemoryInfos(ItemConnectivityMemoryInfo& mem_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Incremental item->item[] connectivity
 */
class ARCANE_MESH_EXPORT IncrementalItemConnectivity
: public IncrementalItemConnectivityBase
{
 private:

  //! For access to _internalNotifySourceItemsAdded().
  friend class IndexedIncrementalItemConnectivityMng;

 public:

  IncrementalItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,
                              const String& aname);
  ~IncrementalItemConnectivity() override;

 public:

  void addConnectedItems(ItemLocalId source_item, Integer nb_item);
  void setConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) override;
  void removeConnectedItems(ItemLocalId source_item) override;
  void addConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override;
  void removeConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override;
  void replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_local_id) override;
  void replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) override;
  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void notifyReadFromDump() override;

 private:

  void _internalNotifySourceItemsAdded(ConstArrayView<Int32> local_ids) override;

 public:

  Integer preAllocatedSize() const final { return m_pre_allocated_size; }
  void setPreAllocatedSize(Integer value) final;

  void dumpStats(std::ostream& out) const override;

  void compactConnectivityList();

 private:

  Int64 m_nb_add = 0;
  Int64 m_nb_remove = 0;
  Int64 m_nb_memcopy = 0;
  Integer m_pre_allocated_size = 0;

 private:

  inline void _increaseIndexList(Int32 lid, Integer size, Int32 target_lid);
  inline Integer _increaseConnectivityList(Int32 new_lid);
  inline Integer _increaseConnectivityList(Int32 new_lid, Integer nb_value);
  inline Integer _computeAllocSize(Integer nb_item);
  void _checkAddNullItem();
  void _resetConnectivityList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Incremental item->item connectivity.
 *
 * This is a specialization of IncrementalItemConnectivity
 * for the case where there is only one connected entity.
 *
 * In this simple case, for an entity with localId() \a lid:
 * - m_connectivity_index[lid] = lid
 * - m_connectivity_nb_item[lid] = 1 (or 0 if no entity has been added yet)
 * - m_connectivity_list[lid] = localId() of the connected entity.
 */
class ARCANE_MESH_EXPORT OneItemIncrementalItemConnectivity
: public IncrementalItemConnectivityBase
{
 private:
 public:

  OneItemIncrementalItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,
                                     const String& aname);
  ~OneItemIncrementalItemConnectivity() override;

 public:

  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void removeConnectedItems(ItemLocalId source_item) override;
  void addConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override;
  void removeConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) override;
  void replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_local_id) override;
  void replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) override;
  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void notifyReadFromDump() override;
  Integer preAllocatedSize() const final { return 1; }
  void setPreAllocatedSize([[maybe_unused]] Int32 value) final {}

 public:

  void dumpStats(std::ostream& out) const override;

  void compactConnectivityList();

 private:

  inline void _checkResizeConnectivityList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
