// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalItemConnectivity.cc                              (C) 2000-2024 */
/*                                                                           */
/* Incremental connectivity of entities.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/IncrementalItemConnectivity.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IIncrementalItemConnectivityInternal.h"

#include "arcane/mesh/IndexedItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor::
IndexedItemConnectivityAccessor(IndexedItemConnectivityViewBase view, IItemFamily* target_item_family)
: IndexedItemConnectivityViewBase(view)
, m_item_shared_info(target_item_family->_internalApi()->commonItemSharedInfo())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor::
IndexedItemConnectivityAccessor(IIncrementalItemConnectivity* connectivity)
: m_item_shared_info(connectivity->targetFamily()->_internalApi()->commonItemSharedInfo())
{
  auto* ptr = dynamic_cast<mesh::IncrementalItemConnectivityBase*>(connectivity);
  if (ptr)
    IndexedItemConnectivityViewBase::set(ptr->connectivityView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractIncrementalItemConnectivity::
AbstractIncrementalItemConnectivity(IItemFamily* source_family,
                                    IItemFamily* target_family,
                                    const String& connectivity_name)
: TraceAccessor(source_family->traceMng())
, m_source_family(source_family)
, m_target_family(target_family)
, m_name(connectivity_name)
{
  m_families.add(m_source_family);
  m_families.add(m_target_family);

  //TODO: these references must be removed upon destruction.
  source_family->_internalApi()->addSourceConnectivity(this);
  target_family->_internalApi()->addTargetConnectivity(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIncrementalItemSourceConnectivity> AbstractIncrementalItemConnectivity::
toSourceReference()
{
  return Arccore::makeRef<IIncrementalItemSourceConnectivity>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIncrementalItemTargetConnectivity> AbstractIncrementalItemConnectivity::
toTargetReference()
{
  return Arccore::makeRef<IIncrementalItemTargetConnectivity>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IncrementalItemConnectivityContainer
{
 public:

  IncrementalItemConnectivityContainer(IMesh* mesh, const String& var_name)
  : m_var_name(var_name)
  , m_connectivity_nb_item_variable(VariableBuildInfo(mesh, var_name + "Nb", IVariable::PPrivate))
  , m_connectivity_index_variable(VariableBuildInfo(mesh, var_name + "Index", IVariable::PPrivate))
  , m_connectivity_list_variable(VariableBuildInfo(mesh, var_name + "List", IVariable::PPrivate))
  , m_connectivity_nb_item_array(m_connectivity_nb_item_variable._internalTrueData()->_internalDeprecatedValue())
  , m_connectivity_index_array(m_connectivity_index_variable._internalTrueData()->_internalDeprecatedValue())
  , m_connectivity_list_array(m_connectivity_list_variable._internalTrueData()->_internalDeprecatedValue())
  {
    // Adds a tag to indicate that these variables are associated with connectivity.
    // For now, this is only used for display statistics.

    String tag_name = "ArcaneConnectivity";
    m_connectivity_nb_item_variable.addTag(tag_name, "1");
    m_connectivity_index_variable.addTag(tag_name, "1");
    m_connectivity_list_variable.addTag(tag_name, "1");
  }

  String m_var_name;

  VariableArrayInt32 m_connectivity_nb_item_variable;
  VariableArrayInt32 m_connectivity_index_variable;
  VariableArrayInt32 m_connectivity_list_variable;

  VariableArrayInt32::ContainerType& m_connectivity_nb_item_array;
  VariableArrayInt32::ContainerType& m_connectivity_index_array;
  VariableArrayInt32::ContainerType& m_connectivity_list_array;

  ObserverPool m_observers;

  /*!
   * \brief Maximum number of connected items.
   *
   * This is an upper bound of the maximum number of connected items.
   * For performance reasons, this value is not updated
   * if items are removed.
   */
  Int32 m_max_nb_item = 0;

 public:

  Integer size() const { return m_connectivity_nb_item_array.size(); }

  bool isAllocated() const { return size() > 0; }

  void _checkResize(Int32 lid)
  {
    //TODO: reuse the code from ItemFamily::_setUniqueId().
    Integer size = m_connectivity_nb_item_array.size();
    Integer wanted_size = lid + 1;
    if (wanted_size < size)
      return;
    Integer capacity = m_connectivity_nb_item_array.capacity();
    if (wanted_size < capacity) {
      // No need to increase capacity.
    }
    else {
      Integer reserve_size = 1000;
      while (lid > reserve_size) {
        reserve_size *= 2;
      }
      m_connectivity_nb_item_array.reserve(reserve_size);
      m_connectivity_index_array.reserve(reserve_size);
    }
    m_connectivity_nb_item_array.resize(wanted_size);
    m_connectivity_index_array.resize(wanted_size);
  }

  void reserveForItems(Int32 capacity)
  {
    m_connectivity_nb_item_array.reserve(capacity);
    m_connectivity_index_array.reserve(capacity);
  }

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IncrementalItemConnectivityBase::InternalApi
: public IIncrementalItemConnectivityInternal
{
 public:

  explicit InternalApi(IncrementalItemConnectivityBase* v)
  : m_internal_api(v)
  {}

 public:

  void shrinkMemory() override { return m_internal_api->_shrinkMemory(); }
  void addMemoryInfos(ItemConnectivityMemoryInfo& mem_info) override
  {
    m_internal_api->_addMemoryInfos(mem_info);
  }

 private:

  IncrementalItemConnectivityBase* m_internal_api = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivityBase::
IncrementalItemConnectivityBase(IItemFamily* source_family, IItemFamily* target_family,
                                const String& aname)
: AbstractIncrementalItemConnectivity(source_family, target_family, aname)
, m_internal_api(std::make_unique<InternalApi>(this))
{
  StringBuilder var_name("Connectivity");
  var_name += aname;
  var_name += source_family->name();
  var_name += target_family->name();

  IMesh* mesh = source_family->mesh();
  m_p = new IncrementalItemConnectivityContainer(mesh, var_name);

  using ThatClass = IncrementalItemConnectivityBase;
  // Get read events to indicate that the views must be updated.
  m_p->m_observers.addObserver(this, &ThatClass::_notifyConnectivityNbItemChangedFromObservable,
                               m_p->m_connectivity_nb_item_variable.variable()->readObservable());

  m_p->m_observers.addObserver(this, &ThatClass::_notifyConnectivityIndexChanged,
                               m_p->m_connectivity_index_variable.variable()->readObservable());

  m_p->m_observers.addObserver(this, &ThatClass::_notifyConnectivityListChanged,
                               m_p->m_connectivity_list_variable.variable()->readObservable());

  // Update the views from the associated arrays.
  // This must be done whenever the size of an array changes because then
  // the array might be reallocated and thus the associated view might become invalid.
  _notifyConnectivityListChanged();
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChangedFromObservable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivityBase::
~IncrementalItemConnectivityBase()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity)
{
  if (n <= 0)
    return;

  m_p->reserveForItems(n);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();

  if (pre_alloc_connectivity) {
    Int32 pre_alloc_size = preAllocatedSize();
    if (pre_alloc_size > 0) {
      m_p->m_connectivity_list_array.reserve(n * pre_alloc_size);
      _notifyConnectivityListChanged();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityListChanged()
{
  m_connectivity_list = m_p->m_connectivity_list_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->_setConnectivityList(m_item_connectivity_index, m_connectivity_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityIndexChanged()
{
  m_connectivity_index = m_p->m_connectivity_index_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->_setConnectivityIndex(m_item_connectivity_index, m_connectivity_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityNbItemChanged()
{
  m_connectivity_nb_item = m_p->m_connectivity_nb_item_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->_setConnectivityNbItem(m_item_connectivity_index, m_connectivity_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_setMaxNbConnectedItemsInConnectivityList()
{
  if (m_item_connectivity_list)
    m_item_connectivity_list->_setMaxNbConnectedItem(m_item_connectivity_index, m_p->m_max_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Method called when the number of entities is modified externally,
 * for example during recovery or after a rollback.
 */
void IncrementalItemConnectivityBase::
_notifyConnectivityNbItemChangedFromObservable()
{
  _notifyConnectivityNbItemChanged();
  _computeMaxNbConnectedItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_setNewMaxNbConnectedItems(Int32 new_max)
{
  if (new_max > m_p->m_max_nb_item) {
    m_p->m_max_nb_item = new_max;
    _setMaxNbConnectedItemsInConnectivityList();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_computeMaxNbConnectedItem()
{
  // Force reset to ensure it is updated
  m_p->m_max_nb_item = -1;
  Int32 max_nb_item = 0;
  for (Int32 x : m_connectivity_nb_item)
    if (x > max_nb_item)
      max_nb_item = x;
  _setNewMaxNbConnectedItems(max_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 IncrementalItemConnectivityBase::
maxNbConnectedItem() const
{
  return m_p->m_max_nb_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indicates that this connectivity is associated with one of the connectivities
 * of ItemInternal.
 *
 * This allows direct updating of the \a ilist structure whenever
 * this connectivity is modified.
 */
void IncrementalItemConnectivityBase::
setItemConnectivityList(ItemInternalConnectivityList* ilist, Int32 index)
{
  info(4) << "setItemConnectivityList name=" << name() << " ilist=" << ilist << " index=" << index;
  m_item_connectivity_list = ilist;
  m_item_connectivity_index = index;
  _notifyConnectivityListChanged();
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();
  _setMaxNbConnectedItemsInConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
{
  if (m_p->isAllocated()) {
    m_p->m_connectivity_nb_item_variable.variable()->compact(new_to_old_ids);
    m_p->m_connectivity_index_variable.variable()->compact(new_to_old_ids);
    _notifyConnectivityNbItemChanged();
    _notifyConnectivityIndexChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids)
{
  Int32ArrayView ids = m_connectivity_list;
  const Integer n = ids.size();
  for (Integer i = 0; i < n; ++i)
    if (ids[i] != NULL_ITEM_LOCAL_ID)
      ids[i] = old_to_new_ids[ids[i]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView IncrementalItemConnectivityBase::
_connectedItems(ItemLocalId item, ConnectivityItemVector& con_items) const
{
  return con_items.resizeAndCopy(_connectedItemsLocalId(item));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemConnectivityContainerView IncrementalItemConnectivityBase::
connectivityContainerView() const
{
  return { m_connectivity_list, m_connectivity_index, m_connectivity_nb_item };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityViewBase IncrementalItemConnectivityBase::
connectivityView() const
{
  return { connectivityContainerView(), _sourceFamily()->itemKind(), _targetFamily()->itemKind() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor IncrementalItemConnectivityBase::
connectivityAccessor() const
{
  return IndexedItemConnectivityAccessor(connectivityView(), _targetFamily());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIncrementalItemConnectivityInternal* IncrementalItemConnectivityBase::
_internalApi()
{
  return m_internal_api.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
dumpInfos()
{
  info() << "Infos index=" << m_connectivity_index;
  info() << "Infos nb_item=" << m_connectivity_nb_item;
  info() << "Infos list=" << m_connectivity_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivity::
IncrementalItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,
                            const String& aname)
: IncrementalItemConnectivityBase(source_family, target_family, aname)
, m_nb_add(0)
, m_nb_remove(0)
, m_nb_memcopy(0)
, m_pre_allocated_size(0)
{
  m_pre_allocated_size = _sourceFamily()->properties()->getIntegerWithDefault(name() + "PreallocSize", 0);
  info(4) << "PreallocSize1 var=" << m_p->m_var_name << " v=" << m_pre_allocated_size;

  // Checks if the null entity needs to be added at the beginning of the list.
  _checkAddNullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivity::
~IncrementalItemConnectivity()
{
  info(4) << " connectivity name=" << name()
          << " prealloc_size=" << m_pre_allocated_size
          << " nb_add=" << m_nb_add
          << " nb_remove=" << m_nb_remove
          << " nb_memcopy=" << m_nb_memcopy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer IncrementalItemConnectivity::
_increaseConnectivityList(Int32 new_lid)
{
  Integer pos_in_list = m_connectivity_list.size();
  m_p->m_connectivity_list_array.add(new_lid);
  _notifyConnectivityListChanged();
  return pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer IncrementalItemConnectivity::
_increaseConnectivityList(Int32 new_lid, Integer nb_value)
{
  Integer pos_in_list = m_connectivity_list.size();
  m_p->m_connectivity_list_array.addRange(new_lid, nb_value);
  _notifyConnectivityListChanged();
  return pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_resetConnectivityList()
{
  m_p->m_connectivity_list_array.clear();
  _notifyConnectivityListChanged();
  _checkAddNullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_increaseIndexList(Int32 lid, Integer size, Int32 target_lid)
{
  Integer added_range = (m_pre_allocated_size > 0) ? m_pre_allocated_size : 1;
  ++m_nb_memcopy;
  Integer pos_in_index = m_connectivity_index[lid];
  Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID, size + added_range);
  ArrayView<Int32> current_list(size, &(m_connectivity_list[pos_in_index]));
  ArrayView<Int32> new_list(size + 1, &(m_connectivity_list[new_pos_in_list]));
  new_list.copy(current_list);
  // Adds the new entity to the end of the connectivity list
  // TODO: look into sorting by increasing uid().
  new_list[size] = target_lid;
  m_connectivity_index[lid] = new_pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
addConnectedItem(ItemLocalId source_item, ItemLocalId target_item)
{
  ++m_nb_add;
  const Int32 lid = source_item.localId();
  const Int32 target_lid = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  // Adds a connected item.
  // For now, the functionality is basic.
  // The items are always added to the end of m_p->m_connectivity_list.
  // If there are none, it is enough to add to the end.
  // If there are already some, it is necessary to allocate at the end of
  // m_p->m_connectivity_list enough \a size+1 elements and
  // we copy the previous connectivity to the new location.
  // Naturally, over time the list will always grow
  // because the gaps are not reused.
  if (m_pre_allocated_size != 0) {
    // In case of preallocation, we allocate in blocks of size 'm_pre_allocated_size'.
    // We must therefore reallocate if the size is a multiple of m_pre_allocated_size
    if (size == 0) {
      Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID, m_pre_allocated_size);
      m_connectivity_index[lid] = new_pos_in_list;
      m_connectivity_list[new_pos_in_list] = target_lid;
    }
    else {
      if (size < m_pre_allocated_size || (size % m_pre_allocated_size) != 0) {
        Integer index = m_connectivity_index[lid];
        m_connectivity_list[index + size] = target_lid;
      }
      else {
        _increaseIndexList(lid, size, target_lid);
      }
    }
  }
  else {
    if (size == 0) {
      Integer new_pos_in_list = _increaseConnectivityList(target_lid);
      m_connectivity_index[lid] = new_pos_in_list;
    }
    else {
      _increaseIndexList(lid, size, target_lid);
    }
  }
  ++(m_connectivity_nb_item[lid]);
  _setNewMaxNbConnectedItems(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer IncrementalItemConnectivity::
_computeAllocSize(Integer nb_item)
{
  if (m_pre_allocated_size != 0) {
    // Allocates a multiple of \a m_pre_allocated_size
    Integer alloc_size = nb_item / m_pre_allocated_size;
    if (alloc_size == 0)
      return m_pre_allocated_size;
    if ((nb_item % m_pre_allocated_size) == 0)
      return nb_item;
    return m_pre_allocated_size * (alloc_size + 1);
  }
  return nb_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
addConnectedItems(ItemLocalId source_item, Integer nb_item)
{
  const Int32 lid = source_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size != 0)
    ARCANE_FATAL("source_item already have connected items");
  Integer alloc_size = _computeAllocSize(nb_item);
  Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID, alloc_size);
  m_connectivity_index[lid] = new_pos_in_list;
  m_connectivity_nb_item[lid] += nb_item;
  _setNewMaxNbConnectedItems(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
setConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids)
{
  removeConnectedItems(source_item);
  addConnectedItems(source_item, target_local_ids.size());
  replaceConnectedItems(source_item, target_local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
removeConnectedItems(ItemLocalId source_item)
{
  Int32 lid = source_item.localId();
  m_connectivity_nb_item[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
removeConnectedItem(ItemLocalId source_item, ItemLocalId target_item)
{
  ++m_nb_remove;
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  Int32* items = &(m_connectivity_list[m_connectivity_index[lid]]);
  mesh_utils::removeItemAndKeepOrder(Int32ArrayView(size, items), target_lid);
  --(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  ARCANE_CHECK_AT(index, m_connectivity_nb_item[lid]);
  m_connectivity_list[m_connectivity_index[lid] + index] = target_lid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids)
{
  Int32 lid = source_item.localId();
  Int32 n = target_local_ids.size();
  if (n > 0) {
    ARCANE_CHECK_AT(n - 1, m_connectivity_nb_item[lid]);
    const Int32 pos = m_connectivity_index[lid];
    for (Integer i = 0; i < n; ++i)
      m_connectivity_list[pos + i] = target_local_ids[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IncrementalItemConnectivity::
hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const
{
  bool has_connection = false;
  auto connected_items = _connectedItemsLocalId(source_item);
  if (std::find(connected_items.begin(), connected_items.end(), target_local_id) != connected_items.end())
    has_connection = true;
  return has_connection;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
notifySourceItemAdded(ItemLocalId item)
{
  Int32 lid = item.localId();
  m_p->_checkResize(lid);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();

  m_connectivity_nb_item[lid] = 0;
  m_connectivity_index[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_internalNotifySourceItemsAdded(ConstArrayView<Int32> local_ids)
{
  // Pre-calculates the maximum of the local_ids for resizing.
  Int32 nb_item = local_ids.size();
  if (nb_item <= 0)
    return;
  Int32 max_lid = local_ids[0];
  for (Int32 lid : local_ids)
    max_lid = math::max(max_lid, lid);

  m_p->_checkResize(max_lid);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();

  for (Int32 lid : local_ids) {
    m_connectivity_nb_item[lid] = 0;
    m_connectivity_index[lid] = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
notifyReadFromDump()
{
  m_pre_allocated_size = _sourceFamily()->properties()->getIntegerWithDefault(name() + "PreallocSize", 0);
  info(4) << "PreallocSize2 var=" << m_p->m_var_name << " v=" << m_pre_allocated_size;

  // There is practically nothing to do for the variables because the views are correctly updated via the observables on the variables.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
setPreAllocatedSize(Integer prealloc_size)
{
  if (m_pre_allocated_size < 0)
    throw ArgumentException(A_FUNCINFO,
                            String::format("Invalid prealloc_size v={0}",
                                           prealloc_size));

  // Does nothing if we have already allocated entities, otherwise it would make
  // the allocations inconsistent.
  // NOTE: we could allow it, but that would require rebuilding
  // the connectivity indices. A call to compactConnectivityList()
  // would suffice.
  if (m_connectivity_nb_item.size() != 0)
    return;

  m_pre_allocated_size = prealloc_size;
  _sourceFamily()->properties()->setInteger(name() + "PreallocSize", prealloc_size);

  // Even if there are no entities, m_p->m_connectivity_list_array is not
  // empty because we called _checkkAddNulItem() in the constructor. We must
  // now reallocate it with the new pre-allocation value.
  _resetConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
dumpStats(std::ostream& out) const
{
  Int64 mem1 = m_p->m_connectivity_list_array.capacity();
  Int64 mem2 = m_p->m_connectivity_index_array.capacity();
  Int64 mem3 = m_p->m_connectivity_nb_item_array.capacity();
  Int64 allocated_size = mem1 + mem2 + mem3;
  allocated_size *= sizeof(Int32);

  out << " connectiviy name=" << name()
      << " prealloc_size=" << m_pre_allocated_size
      << " nb_add=" << m_nb_add
      << " nb_remove=" << m_nb_remove
      << " nb_memcopy=" << m_nb_memcopy
      << " list_size=" << m_connectivity_list.size()
      << " list_capacity=" << mem1
      << " index_size=" << m_connectivity_index.size()
      << " index_capacity=" << mem2
      << " nb_item_size=" << m_connectivity_nb_item.size()
      << " nb_item_capacity=" << mem3
      << " allocated_size=" << allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_shrinkMemory()
{
  m_p->m_connectivity_list_array.shrink();
  m_p->m_connectivity_index_array.shrink();
  m_p->m_connectivity_nb_item_array.shrink();
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();
  _notifyConnectivityListChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_addMemoryInfos(ItemConnectivityMemoryInfo& mem_info)
{
  Int64 s1 = m_p->m_connectivity_list_array.size();
  Int64 s2 = m_p->m_connectivity_index_array.size();
  Int64 s3 = m_p->m_connectivity_nb_item_array.size();
  mem_info.m_total_size += s1 + s2 + s3;

  Int64 c1 = m_p->m_connectivity_list_array.capacity();
  Int64 c2 = m_p->m_connectivity_index_array.capacity();
  Int64 c3 = m_p->m_connectivity_nb_item_array.capacity();
  mem_info.m_total_capacity += c1 + c2 + c3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_checkAddNullItem()
{
  // If the connectivity list is empty, create an element
  // (or several if m_pre_allocated_size>0) to contain the null entity.
  // This allows retrieving the list of connectivities for an entity even if
  // it is empty.
  if (m_connectivity_list.size() == 0) {
    if (m_pre_allocated_size > 0) {
      _increaseConnectivityList(NULL_ITEM_LOCAL_ID, m_pre_allocated_size);
    }
    else {
      _increaseConnectivityList(NULL_ITEM_LOCAL_ID);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the connectivity list.
 *
 * The current implementation is quite simple:
 * - Copies the current list into a temporary array.
 * - Clears the current list.
 * - Recopies the useful values from the temporary array into the list.
 *
 * \note Calling this method assumes that the source family entities are compacted.
 */
void IncrementalItemConnectivity::
compactConnectivityList()
{
  info(4) << "Begin Compacting IncrementalItemConnectivity name=" << name()
          << " new_size=" << m_connectivity_list.size()
          << " prealloc_size=" << m_pre_allocated_size;
  // TODO: try to find a way to only compact if
  // it is necessary. One way would be to count the number of calls to
  // _increaseIndexList() since the last compaction.
  UniqueArray<Int32> old_connectivity_list(m_connectivity_list);
  Integer old_size = old_connectivity_list.size();
  Integer nb_item = m_connectivity_nb_item.size();
  m_p->m_connectivity_list_array.clear();
  _notifyConnectivityListChanged();
  _checkAddNullItem();
  Integer new_pos_in_list = m_p->m_connectivity_list_array.size();
  Int32 pre_allocated_size = m_pre_allocated_size;
  for (Integer i = 0; i < nb_item; ++i) {
    Int32 lid = i;
    Int32 nb = m_connectivity_nb_item[lid];
    if (nb == 0) {
      m_connectivity_index[lid] = 0;
      continue;
    }
    Int32 index = m_connectivity_index[lid];
    Int32ConstArrayView con_list(nb, old_connectivity_list.data() + index);
    Integer alloc_size = _computeAllocSize(nb);
    m_connectivity_index[lid] = new_pos_in_list;
    new_pos_in_list += alloc_size;
    //info() << "NEW_POS_IN_LIST=" << new_pos_in_list << " nb=" << nb << " alloc_size=" << alloc_size;
    // Checks that the position is indeed a multiple of pre_allocated_size.
    if (pre_allocated_size != 0) {
      Int32 pos_modulo = new_pos_in_list % pre_allocated_size;
      if (pos_modulo != 0)
        ARCANE_FATAL("Bad position i={0} pos={1} pre_alloc_size={2} modulo={3}",
                     i, new_pos_in_list, pre_allocated_size, pos_modulo);
    }
    m_p->m_connectivity_list_array.addRange(con_list);
    // If pre-allocation, fill the remaining elements with the null entity..
    if (alloc_size != nb)
      m_p->m_connectivity_list_array.addRange(NULL_ITEM_LOCAL_ID, alloc_size - nb);
    if (m_pre_allocated_size == 0 && nb == 0)
      m_connectivity_index[lid] = 0;
  }
  _notifyConnectivityListChanged();
  _computeMaxNbConnectedItem();
  info(4) << "Compacting IncrementalItemConnectivity name=" << name()
          << " nb_item=" << nb_item << " old_size=" << old_size
          << " new_size=" << m_connectivity_list.size()
          << " prealloc_size=" << m_pre_allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneItemIncrementalItemConnectivity::
OneItemIncrementalItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,
                                   const String& aname)
: IncrementalItemConnectivityBase(source_family, target_family, aname)
{
  info(4) << "Using fixed OneItem connectivity for name=" << name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneItemIncrementalItemConnectivity::
~OneItemIncrementalItemConnectivity()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
addConnectedItem(ItemLocalId source_item, ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size != 0)
    ARCANE_FATAL("source_item already have connected items");
  Int32 target_lid = target_item.localId();
  m_connectivity_list[lid] = target_lid;
  m_connectivity_nb_item[lid] = 1;
  _setNewMaxNbConnectedItems(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
removeConnectedItems(ItemLocalId source_item)
{
  Int32 lid = source_item.localId();
  m_connectivity_nb_item[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
removeConnectedItem(ItemLocalId source_item, ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Int32 target_local_id = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size != 1)
    ARCANE_FATAL("source_item has no connected item");
  Int32 target_lid = m_connectivity_list[lid];
  if (target_lid != target_local_id)
    ARCANE_FATAL("source_item is not connected to item with wanted_lid={0} current_lid={1}",
                 target_local_id, target_lid);
  m_connectivity_nb_item[lid] = 0;
  m_connectivity_list[lid] = NULL_ITEM_LOCAL_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_item)
{
  if (index != 0)
    ARCANE_FATAL("index has to be '0'");
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  m_connectivity_list[lid] = target_lid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids)
{
  Int32 lid = source_item.localId();
  Integer n = target_local_ids.size();
  if (n == 0)
    return;
  if (n != 1)
    ARCANE_FATAL("Invalid size for target_list. value={0} (expected 1)", n);
  m_connectivity_list[lid] = target_local_ids[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool OneItemIncrementalItemConnectivity::
hasConnectedItem(ItemLocalId source_item,
                 ItemLocalId target_local_id) const
{
  if (m_connectivity_list[source_item.localId()] == target_local_id.localId())
    return true;
  else
    return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
_checkResizeConnectivityList()
{
  // Resizes the connectivity list with the same number of elements
  // as the number of entities.
  Integer wanted_size = m_connectivity_nb_item.size();
  Integer list_size = m_connectivity_list.size();
  if (list_size == wanted_size)
    return;
  Integer capacity = m_p->m_connectivity_list_array.capacity();
  if (wanted_size >= capacity) {
    m_p->m_connectivity_list_array.reserve(m_p->m_connectivity_nb_item_array.capacity());
  }
  m_p->m_connectivity_list_array.resize(wanted_size);
  _notifyConnectivityListChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifySourceItemAdded(ItemLocalId item)
{
  Int32 lid = item.localId();
  m_p->_checkResize(lid);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();
  _checkResizeConnectivityList();

  m_connectivity_nb_item[lid] = 0;
  m_connectivity_index[lid] = lid;
  m_connectivity_list[lid] = NULL_ITEM_LOCAL_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
{
  // For this implementation, the indices should not be updated
  // because otherwise we won't have m_connectivity_index[lid] = lid.
  // TODO: since m_connectivity_nb_item is originally 1 everywhere, it's
  // not useful to do it on this variable either, but
  // since there might be entities for which nb_item is 0 if
  // no connected entity was added, it is better to perform the compaction.

  m_p->m_connectivity_nb_item_variable.variable()->compact(new_to_old_ids);
  _notifyConnectivityNbItemChanged();

  // Since with this implementation the connectivity list is indexed
  // by the localId() of the source entity, it must be compacted
  // m_p->m_connectivity_list_variable.
  m_p->m_connectivity_list_variable.variable()->compact(new_to_old_ids);
  _notifyConnectivityListChanged();

  // Does not compact the indices but still updates the size
  // of the array.
  m_p->m_connectivity_index_array.resize(m_connectivity_nb_item.size());
  _notifyConnectivityIndexChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifyReadFromDump()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
dumpStats(std::ostream& out) const
{
  size_t allocated_size = m_p->m_connectivity_list_array.capacity() + m_p->m_connectivity_index_array.capacity() + m_p->m_connectivity_nb_item_array.capacity();
  allocated_size *= sizeof(Int32);

  out << " connectiviy name=" << name()
      << " list_size=" << m_connectivity_list.size()
      << " index_size=" << m_connectivity_index.size()
      << " nb_item_size=" << m_connectivity_nb_item.size()
      << " allocated_size=" << allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
compactConnectivityList()
{
  _computeMaxNbConnectedItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
