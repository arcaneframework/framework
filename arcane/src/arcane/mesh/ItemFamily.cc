// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamily.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Mesh info for a given type of entity.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CStringUtils.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/ItemPairGroupImpl.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/ItemInternalSortFunction.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ItemFamilyCompactInfos.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshCompacter.h"
#include "arcane/core/IMeshCompactMng.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IIncrementalItemConnectivityInternal.h"
#include "arcane/core/datatype/IDataOperation.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemSharedInfoList.h"
#include "arcane/mesh/ItemConnectivityInfo.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"
#include "arcane/mesh/DynamicMeshKindInfos.h"

#include "arcane/core/parallel/GhostItemsVariableParallelOperation.h"
#include "arcane/core/parallel/IStat.h"

#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IItemConnectivityMng.h"
#include "arcane/core/IItemFamilyPolicyMng.h"

#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/IndexedItemConnectivityView.h"

#include "arcane/mesh/ItemProperty.h"
#include "arcane/mesh/ItemData.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

  template <typename DataType> void
  _offsetArrayByOne(Array<DataType>* array)
  {
    Array<DataType>& v = *array;
    MeshUtils::checkResizeArray(v, v.size() + 1, false);
    Int32 n = v.size();
    for (Int32 i = (n - 1); i >= 1; --i)
      v[i] = v[i - 1];
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily::InternalApi
: public IItemFamilyInternal
{
 public:

  explicit InternalApi(ItemFamily* family)
  : m_family(family)
  {}

 public:

  ItemInternalConnectivityList* unstructuredItemInternalConnectivityList() override
  {
    return m_family->_unstructuredItemInternalConnectivityList();
  }
  IItemFamilyTopologyModifier* topologyModifier() override
  {
    return m_family->_topologyModifier();
  }
  ItemSharedInfo* commonItemSharedInfo() override
  {
    return m_family->commonItemSharedInfo();
  }
  void addSourceConnectivity(IIncrementalItemSourceConnectivity* connectivity) override
  {
    m_family->_addSourceConnectivity(connectivity);
  }
  void addTargetConnectivity(IIncrementalItemTargetConnectivity* connectivity) override
  {
    m_family->_addTargetConnectivity(connectivity);
  }
  void endAllocate() override
  {
    return m_family->_endAllocate();
  }
  void notifyEndUpdateFromMesh() override
  {
    return m_family->_notifyEndUpdateFromMesh();
  }
  void addVariable(IVariable* var) override
  {
    return m_family->_addVariable(var);
  }
  void removeVariable(IVariable* var) override
  {
    return m_family->_removeVariable(var);
  }
  void resizeVariables(bool force_resize) override
  {
    m_family->_resizeShMemVariables();
    m_family->_resizeVariables(force_resize);
  }

 private:

  ItemFamily* m_family = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily::Variables
{
 public:

  Variables(IMesh* mesh,
            const String& family_name,
            eItemKind item_kind,
            const String& shared_data_name,
            const String& unique_ids_name,
            const String& items_owner_name,
            const String& items_flags_name,
            const String& items_type_id_name,
            const String& items_nb_parent_name,
            const String& groups_name,
            const String& current_id_name,
            const String& new_owner_name,
            const String& parent_mesh_name,
            const String& parent_family_name,
            const String& parent_family_depth_name,
            const String& child_meshes_name,
            const String& child_families_name)
  : m_items_shared_data_index(VariableBuildInfo(mesh, shared_data_name, IVariable::PPrivate))
  , m_items_unique_id(VariableBuildInfo(mesh, unique_ids_name, IVariable::PPrivate))
  , m_items_owner(VariableBuildInfo(mesh, items_owner_name, IVariable::PPrivate))
  , m_items_flags(VariableBuildInfo(mesh, items_flags_name, IVariable::PPrivate))
  , m_items_type_id(VariableBuildInfo(mesh, items_type_id_name, IVariable::PPrivate))
  , m_items_nb_parent(VariableBuildInfo(mesh, items_nb_parent_name, IVariable::PPrivate))
  , m_groups_name(VariableBuildInfo(mesh, groups_name))
  , m_current_id(VariableBuildInfo(mesh, current_id_name))
  , m_items_new_owner(VariableBuildInfo(mesh, new_owner_name, family_name, IVariable::PNoDump | IVariable::PSubDomainDepend | IVariable::PExecutionDepend), item_kind)
  , m_parent_mesh_name(VariableBuildInfo(mesh, parent_mesh_name, IVariable::PPrivate))
  , m_parent_family_name(VariableBuildInfo(mesh, parent_family_name, IVariable::PPrivate))
  , m_parent_family_depth(VariableBuildInfo(mesh, parent_family_depth_name, IVariable::PPrivate))
  , m_child_meshes_name(VariableBuildInfo(mesh, child_meshes_name, IVariable::PPrivate))
  , m_child_families_name(VariableBuildInfo(mesh, child_families_name, IVariable::PPrivate))
  {}

 public:

  void setUsed()
  {
    m_items_new_owner.setUsed(true);
  }

 public:

  //! Index in the ItemSharedInfo array for each entity.
  // TODO: use Int16 when the number of ItemSharedInfo is limited to Int16
  VariableArrayInteger m_items_shared_data_index;
  //! Contains the uniqueIds() of the entities in this family
  VariableArrayInt64 m_items_unique_id;
  //! Contains the owner() of the entities in this family
  VariableArrayInt32 m_items_owner;
  //! Contains the flags() of the entities in this family
  VariableArrayInt32 m_items_flags;
  //! Contains the typeId() of the entities in this family
  VariableArrayInt16 m_items_type_id;
  /*!
   * \brief Contains the parent() of the entities in this family.
   *
   * This is only used with sub-meshes and assumes that there is
   * only one parent per entity. If we ever want multiple parents,
   * this variable will need to be an 'Array2'.
   */
  VariableArrayInt32 m_items_nb_parent;
  VariableArrayString m_groups_name;
  VariableScalarInteger m_current_id;
  /*!
   * \brief Contains the owning sub-domain of the entity.
   *
   * This variable is redundant with the ItemInternal owner() field
   * and only has a different value when entities change
   * owners. Therefore, it should not be necessary to allocate
   * it sequentially.
   * \todo Similarly, its value can be recovered during a restart
   * and it should be marked IVariable::PNoDump.
   */
  VariableItemInt32 m_items_new_owner;
  VariableScalarString m_parent_mesh_name;
  VariableScalarString m_parent_family_name;
  VariableScalarInteger m_parent_family_depth;
  VariableArrayString m_child_meshes_name;
  VariableArrayString m_child_families_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemFamily::
_cmpIVariablePtr(const IVariable* a, const IVariable* b)
{
  return CStringUtils::isLess(a->name().localstr(), b->name().localstr());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamily::
ItemFamily(IMesh* mesh, eItemKind ik, const String& name)
: TraceAccessor(mesh->traceMng())
, m_name(name)
, m_mesh(mesh)
, m_internal_api(new InternalApi(this))
, m_sub_domain(mesh->subDomain())
, m_infos(std::make_unique<DynamicMeshKindInfos>(mesh, ik, name))
, m_item_internal_list(mesh->meshItemInternalList())
, m_common_item_shared_info(new ItemSharedInfo(this, m_item_internal_list, &m_item_connectivity_list))
, m_item_shared_infos(new ItemSharedInfoList(this, m_common_item_shared_info))
, m_used_variables(&_cmpIVariablePtr)
, m_used_shmem_variables(&_cmpIVariablePtr)
, m_properties(new Properties(*mesh->properties(), name))
, m_sub_domain_id(mesh->meshPartInfo().partRank())
{
  m_item_connectivity_list.m_items = mesh->meshItemInternalList();
  m_infos->setItemFamily(this);
  m_connectivity_selector_list_by_item_kind.resize(ItemInternalConnectivityList::MAX_ITEM_KIND);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamily::
~ItemFamily()
{
  delete m_topology_modifier;
  delete m_policy_mng;
  delete m_properties;
  delete m_local_connectivity_info;
  delete m_global_connectivity_info;
  delete m_item_sort_function;
  delete m_internal_variables;
  delete m_item_shared_infos;

  for (ItemConnectivitySelector* ics : m_connectivity_selector_list)
    delete ics;

  delete m_common_item_shared_info;
  delete m_internal_api;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ItemFamily::
_variableName(const String& base_name) const
{
  return m_name + base_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
build()
{
  m_item_type_mng = m_mesh->itemTypeMng();

  // Create the topology modifier if it hasn't been done by
  // the derived class.
  if (!m_topology_modifier)
    m_topology_modifier = new AbstractItemFamilyTopologyModifier(this);

  m_full_name = m_mesh->name() + "_" + m_name;

  m_local_connectivity_info = new ItemConnectivityInfo();
  m_global_connectivity_info = new ItemConnectivityInfo();

  IParallelMng* pm = m_mesh->parallelMng();
  if (!pm->isParallel()) {
    m_default_sub_domain_owner = 0;
    m_is_parallel = false;
  }
  else
    m_is_parallel = true;

  // First initialize the infos because this creates the entity groups
  // and it is essential before creating variables on them.
  m_infos->build();

  // Constructs the instance that will contain the variables
  // NOTE: if we change the names here, we must also change them in MeshStats.cc
  // otherwise the statistics will not be reliable.
  {
    String var_unique_ids_name(_variableName("FamilyUniqueIds"));
    String var_owner_name(_variableName("FamilyOwner"));
    String var_flags_name(_variableName("FamilyFlags"));
    String var_typeid_name(_variableName("FamilyItemTypeId"));
    String var_nb_parent_name(_variableName("FamilyItemNbParent"));
    String var_count_name(_variableName("FamilyItemsShared"));
    String var_groups_name(_variableName("FamilyGroupsName"));
    String var_current_id_name(_variableName("FamilyCurrentId"));
    String var_new_owner_name(_variableName("FamilyNewOwnerName"));
    String var_parent_mesh_name(_variableName("ParentMeshName"));
    String var_parent_family_name(_variableName("ParentFamilyName"));
    String var_parent_family_depth_name(_variableName("ParentFamilyDepthName"));
    String var_child_meshes_name(_variableName("ChildMeshesName"));
    String var_child_families_name(_variableName("ChildFamiliesName"));
    m_internal_variables = new Variables(m_mesh, name(), itemKind(), var_count_name,
                                         var_unique_ids_name, var_owner_name,
                                         var_flags_name, var_typeid_name, var_nb_parent_name, var_groups_name,
                                         var_current_id_name, var_new_owner_name,
                                         var_parent_mesh_name, var_parent_family_name,
                                         var_parent_family_depth_name,
                                         var_child_meshes_name,
                                         var_child_families_name);
    // These arrays should not be used to access entities because there is an offset
    // of 1 to manage the null entity. You must use the associated view
    // which is in m_common_item_shared_info.
    m_items_unique_id = &m_internal_variables->m_items_unique_id._internalTrueData()->_internalDeprecatedValue();
    m_items_owner = &m_internal_variables->m_items_owner._internalTrueData()->_internalDeprecatedValue();
    m_items_flags = &m_internal_variables->m_items_flags._internalTrueData()->_internalDeprecatedValue();
    m_items_type_id = &m_internal_variables->m_items_type_id._internalTrueData()->_internalDeprecatedValue();
    m_items_nb_parent = &m_internal_variables->m_items_nb_parent._internalTrueData()->_internalDeprecatedValue();

    // Add notification for view update after an external change
    _addOnSizeChangedObservable(m_internal_variables->m_items_unique_id);
    _addOnSizeChangedObservable(m_internal_variables->m_items_owner);
    _addOnSizeChangedObservable(m_internal_variables->m_items_flags);
    _addOnSizeChangedObservable(m_internal_variables->m_items_type_id);
    _addOnSizeChangedObservable(m_internal_variables->m_items_nb_parent);

    _updateItemViews();
  }

  m_variable_synchronizer = ParallelMngUtils::createSynchronizerRef(pm, this);

  m_item_sort_function = _defaultItemSortFunction();

  {
    String s = platform::getEnvironmentVariable("ARCANE_USE_LEGACY_COMPACT_ITEMS");
    if (s == "TRUE" || s == "1") {
      info() << "WARNING: Using legacy 'compactItem()' without compactReference()'";
      m_use_legacy_compact_item = true;
    }
  }
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ITEMFAMILY_SHRINK_AFTER_ALLOCATE", true))
      m_do_shrink_after_allocate = (v.value() != 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemFamily::
findOneItem(Int64 uid)
{
  return m_infos->findOne(uid);
}
ItemInternalMap& ItemFamily::
itemsMap()
{
  return m_infos->itemsMap();
}
const DynamicMeshKindInfos& ItemFamily::
infos() const
{
  return *m_infos;
}

const DynamicMeshKindInfos& ItemFamily::
_infos() const
{
  return *m_infos;
}

void ItemFamily::
_removeOne(Item item)
{
  // TODO: check in check mode with the new connectivities that the deleted entity
  // does not have connected objects.
  m_infos->removeOne(ItemCompatibility::_itemInternal(item));
}
void ItemFamily::
_detachOne(Item item)
{
  m_infos->detachOne(ItemCompatibility::_itemInternal(item));
}
ItemInternalList ItemFamily::
_itemsInternal()
{
  return m_infos->itemsInternal();
}
ItemInternal* ItemFamily::
_itemInternal(Int32 local_id)
{
  return m_infos->itemInternal(local_id);
}
ItemInternal* ItemFamily::
_allocOne(Int64 unique_id)
{
  return m_infos->allocOne(unique_id);
}
ItemInternal* ItemFamily::
_allocOne(Int64 unique_id, bool& need_alloc)
{
  return m_infos->allocOne(unique_id, need_alloc);
}
ItemInternal* ItemFamily::
_findOrAllocOne(Int64 uid, bool& is_alloc)
{
  return m_infos->findOrAllocOne(uid, is_alloc);
}
void ItemFamily::
_setHasUniqueIdMap(bool v)
{
  m_infos->setHasUniqueIdMap(v);
}
void ItemFamily::
_removeMany(Int32ConstArrayView local_ids)
{
  m_infos->removeMany(local_ids);
}
void ItemFamily::
_removeDetachedOne(Item item)
{
  m_infos->removeDetachedOne(ItemCompatibility::_itemInternal(item));
}
eItemKind ItemFamily::
itemKind() const
{
  return m_infos->kind();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
nbItem() const
{
  return m_infos->nbItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemFamily::
maxLocalId() const
{
  return m_infos->maxUsedLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalList ItemFamily::
itemsInternal()
{
  return m_infos->itemsInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInfoListView ItemFamily::
itemInfoListView()
{
  return ItemInfoListView(m_common_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemFamily::
parentFamily() const
{
  return m_parent_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setParentFamily(IItemFamily* parent)
{
  m_parent_family = parent;
  if (parent == this) // Self-referencing
    m_parent_family_depth = 1;
  else if (!parent) // No parent
    m_parent_family_depth = 0;
  else { // Cross-referencing
    m_parent_family_depth = parent->parentFamilyDepth() + 1;
    m_parent_family->addChildFamily(this);
  }
  ARCANE_ASSERT((m_parent_family_depth < 2), ("Not test if more than one depth level"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
parentFamilyDepth() const
{
  return m_parent_family_depth;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addChildFamily(IItemFamily* family)
{
  ItemFamily* true_family = ARCANE_CHECK_POINTER(dynamic_cast<ItemFamily*>(family));
  m_child_families.add(true_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyCollection ItemFamily::
childFamilies()
{
  IItemFamilyCollection collection = List<IItemFamily*>();
  for (Integer i = 0; i < m_child_families.size(); ++i) {
    collection.add(m_child_families[i]);
  }
  return collection;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableItemInt32& ItemFamily::
itemsNewOwner()
{
  return m_internal_variables->m_items_new_owner;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
checkValid()
{
  _checkValid();
  m_item_shared_infos->checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
checkValidConnectivity()
{
  _checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Method called by the mesh at the end of an IMesh::endUpdate().
 * This method is collective and therefore allows collective operations
 * once mesh modifications are finished.
 */
void ItemFamily::
_notifyEndUpdateFromMesh()
{
  // Recalculate local and global connectivity info for all sub-domains
  _computeConnectivityInfo(m_local_connectivity_info);
  _computeConnectivityInfo(m_global_connectivity_info);
  m_global_connectivity_info->reduce(parallelMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
endUpdate()
{
  _endUpdate(true);
  // TODO: test connectivity but it crashes some test cases
  // (dof, amr2 and mesh_modification). To see if this is normal.
  //if (arcaneIsCheck())
  //checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_endAllocate()
{
  // The variable is not "used" by default because the family is not ready yet.
  // On sub-families, it is enough to filter setUsed at the time of endAllocate
  if (!m_parent_family) {
    m_internal_variables->setUsed();
  }
  if (m_do_shrink_after_allocate)
    _shrinkConnectivityAndPrintInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_shrinkConnectivityAndPrintInfos()
{
  {
    ItemConnectivityMemoryInfo mem_info;
    for (ItemConnectivitySelector* cs : m_connectivity_selector_list) {
      IIncrementalItemConnectivity* c = cs->customConnectivity();
      c->_internalApi()->addMemoryInfos(mem_info);
    }
    const Int64 total_capacity = mem_info.m_total_capacity;
    const Int64 total_size = mem_info.m_total_size;
    Int64 ratio = 100 * (total_capacity - total_size);
    ratio /= (total_size + 1); // Add 1 to avoid division by zero
    const Int64 sizeof_int32 = sizeof(Int32);
    const Int64 mega_byte = 1024 * 1024;
    Int64 capacity_mega_byte = (mem_info.m_total_capacity * sizeof_int32) / mega_byte;
    info() << "MemoryUsed for family name=" << name() << " size=" << mem_info.m_total_size
           << " capacity=" << mem_info.m_total_capacity
           << " capacity (MegaByte)=" << capacity_mega_byte
           << " ratio=" << ratio;
  }
  OStringStream ostr;
  std::ostream& o = ostr();
  o << "Mem=" << platform::getMemoryUsed();
  for (ItemConnectivitySelector* cs : m_connectivity_selector_list) {
    IIncrementalItemConnectivity* c = cs->customConnectivity();
    c->dumpStats(o);
    o << "\n";
    c->_internalApi()->shrinkMemory();
    c->dumpStats(o);
    o << "\n";
  }
  o << "Mem=" << platform::getMemoryUsed();
  info() << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemFamily::
_partialEndUpdate()
{
  bool need_end_update = m_infos->changed();
  info(4) << "ItemFamily::endUpdate() " << fullName() << " need_end_update?=" << need_end_update;
  if (!need_end_update) {
    // Even if no entity is added or removed, if m_need_prepare_dump
    // is true, it means something has changed in the family. In this
    // case, it is preferable to increment the current_id. Otherwise, if we call
    // readFromDump() (for example, following a rollback), the data will not be
    // restored if current_id has not changed in the meantime.
    if (m_need_prepare_dump) {
      _computeConnectivityInfo(m_local_connectivity_info);
      ++m_current_id;
    }
    return true;
  }
  m_item_need_prepare_dump = true;
  _computeConnectivityInfo(m_local_connectivity_info);
  ++m_current_id;

  // Update "external" connectivities
  if (m_connectivity_mng)
    m_connectivity_mng->setModifiedItems(this, m_infos->addedItems(), m_infos->removedItems());
  //
  m_infos->finalizeMeshChanged();

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_endUpdate(bool need_check_remove)
{
  _resizeShMemVariables();
  if (_partialEndUpdate())
    return;

  _resizeVariables(false);
  info(4) << "ItemFamily:endUpdate(): " << fullName()
          << " hashmapsize=" << itemsMap().nbBucket()
          << " nb_group=" << m_item_groups.count();

  _updateGroups(need_check_remove);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
partialEndUpdate()
{
  _partialEndUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
partialEndUpdateGroup(const ItemGroup& group)
{
  _updateGroup(group, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateGroup(ItemGroup group, bool need_check_remove)
{
  // No need to recalculate the group of global entities
  if (group == m_infos->allItems())
    return;

  if (group.internal()->hasComputeFunctor())
    group.invalidate();
  if (need_check_remove) {
    debug(Trace::High) << "Reset SuppressedItems: " << group.name();
    group.internal()->removeSuppressedItems();
  }
  if (arcaneIsCheck())
    group.checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateGroups(bool need_check_remove)
{
  for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
    ItemGroup group = *i;
    _updateGroup(group, need_check_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
partialEndUpdateVariable(IVariable* variable)
{
  _updateVariable(variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateVariable(IVariable* var)
{
  var->resizeFromGroup();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_resizeVariables(bool force_resize)
{
  debug(Trace::High) << "ItemFamily::resizeVariables: name=" << fullName()
                     << " varsize=" << maxLocalId()
                     << " nb_item=" << nbItem()
                     << " currentsize=" << m_current_variable_item_size;
  if (force_resize || (maxLocalId() != m_current_variable_item_size)) {
    info(4) << "ItemFamily::resizeVariables: name=" << fullName()
            << " varsize=" << maxLocalId()
            << " nb_item=" << nbItem()
            << " currentsize=" << m_current_variable_item_size
            << " group_nb_item=" << allItems().size()
            << " nb_var=" << m_used_variables.size();

    m_current_variable_item_size = maxLocalId();

    for (IVariable* var : m_used_variables) {
      _updateVariable(var);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_resizeShMemVariables()
{
  for (IVariable* var : m_used_shmem_variables) {
    _updateVariable(var);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(ArrayView<Int64> ids, bool do_fatal) const
{
  m_infos->itemsUniqueIdToLocalId(ids, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       Int64ConstArrayView unique_ids,
                       bool do_fatal) const
{
  m_infos->itemsUniqueIdToLocalId(local_ids, unique_ids, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       ConstArrayView<ItemUniqueId> unique_ids,
                       bool do_fatal) const
{
  m_infos->itemsUniqueIdToLocalId(local_ids, unique_ids, do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* ItemFamily::
subDomain() const
{
  return m_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* ItemFamily::
traceMng() const
{
  return TraceAccessor::traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* ItemFamily::
mesh() const
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* ItemFamily::
parallelMng() const
{
  return m_mesh->parallelMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
allItems() const
{
  _checkNeedEndUpdate();
  return m_infos->allItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
createGroup(const String& name, Int32ConstArrayView elements, bool do_override)
{
  debug(Trace::High) << "ItemFamily:createGroup(name,Int32ConstArrayView): " << m_name << ":"
                     << " group_name=" << name
                     << " count=" << elements.size()
                     << " override=" << do_override;

  _checkNeedEndUpdate();

  {
    ItemGroup group;
    if (do_override) {
      group = findGroup(name);
      if (group.null())
        group = createGroup(name);
    }
    else {
      group = createGroup(name);
    }
    group.setItems(elements);
    return group;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
createGroup(const String& name)
{
  //#warning SDP CHANGE
  // SDP: checkpointing/recovery problem ...
  {
    ItemGroup g = findGroup(name);
    if (!g.null()) {
      fatal() << "Attempting to create an already existing group '" << name << "'";
    }
  }
  debug() << "ItemFamily:createGroup(name): " << m_name << ":"
          << " name=" << name;
  ItemGroup group(new ItemGroupImpl(this, name));
  _processNewGroup(group);
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
createGroup(const String& name, const ItemGroup& parent, bool do_override)
{
  ItemGroup group = findGroup(name);
  if (!group.null()) {
    if (do_override) {
      if (group.internal()->parentGroup() != parent)
        fatal() << "Group already existing but with a different parent";
      if (group == parent)
        fatal() << "A group can not be its own parent name=" << name;
      return group;
    }
    fatal() << "Attempting to create an already existing group '" << name << "'";
  }
  if (parent.null())
    fatal() << "Attempting to create a group '" << name << "' with no parent.";
  debug() << "ItemFamily:createGroup(name,parent): " << m_name << ":"
          << " name=" << name
          << " parent=" << parent.name();
  group = ItemGroup(new ItemGroupImpl(this, parent.internal(), name));
  _processNewGroup(group);
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
destroyGroups()
{
  _invalidateComputedGroups();
  ItemGroupList current_groups(m_item_groups.clone());
  m_item_groups.clear();
  for (ItemGroupList::Enumerator i(current_groups); ++i;) {
    ItemGroup group(*i);
    if (group.isAllItems())
      m_item_groups.add(group);
    else
      group.internal()->destroy();
  }
  allItems().internal()->destroy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
notifyItemsOwnerChanged()
{
  debug() << "ItemFamily::notifyItemsOwnerChanged()";
  for (ItemGroup group : m_item_groups) {
    if (m_is_parallel && group.internal()->hasComputeFunctor())
      group.invalidate();
  }

  // Propagate changes to child families
  for (Integer i = 0; i < m_child_families.size(); ++i) {
    IItemFamily* family = m_child_families[i];
    ItemInternalArrayView items(family->itemsInternal());
    for (Integer z = 0, zs = items.size(); z < zs; ++z) {
      impl::MutableItemBase item(items[z]);
      if (item.isSuppressed())
        continue;
      Item parent_item = item.parentBase(0);
      ARCANE_ASSERT((parent_item.uniqueId() == item.uniqueId()), ("Inconsistent parent uid"));
      item.setOwner(parent_item.owner(), m_sub_domain_id);
    }
    family->notifyItemsOwnerChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_processNewGroup(ItemGroup group)
{
  // Checks that the group has the correct type
  if (group.itemKind() != itemKind()) {
    ARCANE_FATAL("Incoherent family name={0} wanted={1} current={2}",
                 fullName(), itemKind(), group.itemKind());
  }
  m_item_groups.add(group);
  m_need_prepare_dump = true;
  // In sequential mode, all groups are owned groups
  // TODO: look into removing the test with 'm_is_parallel' but
  // if we do that it crashes some tests with aleph_kappa.
  if (!m_is_parallel && m_mesh->meshPartInfo().nbPart() == 1)
    group.setOwn(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupCollection ItemFamily::
groups() const
{
  return m_item_groups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
findGroup(const String& name) const
{
  for (const ItemGroup& group : m_item_groups) {
    if (group.name() == name)
      return group;
  }
  return ItemGroup();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
findGroup(const String& name, bool create_if_needed)
{
  ItemGroup group = findGroup(name);
  if (group.null()) {
    if (create_if_needed)
      group = createGroup(name);
  }
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
notifyItemsUniqueIdChanged()
{
  itemsMap().notifyUniqueIdsChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_checkNeedEndUpdate() const
{
  if (m_infos->changed())
    ARCANE_FATAL("missing call to endUpdate() after modification");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
prepareForDump()
{
  info(4) << "ItemFamily::prepareForDump(): " << fullName()
          << " need=" << m_need_prepare_dump
          << " item-need=" << m_item_need_prepare_dump
          << " m_item_shared_infos->hasChanged()=" << m_item_shared_infos->hasChanged()
          << " nb_item=" << m_infos->nbItem();

  {
    auto* p = m_properties;
    p->setInt32("dump-version", 0x0307);
    p->setInt32("nb-item", m_infos->nbItem());
    p->setInt32("current-change-id", m_current_id);
  }

  // TODO: add verification flag if necessary
  if (m_item_need_prepare_dump || m_item_shared_infos->hasChanged()) {
    info(4) << "Prepare for dump:2: name=" << m_name << " nb_alloc=" << m_nb_allocate_info
            << " uid_size=" << m_items_unique_id->size() << " cap=" << m_items_unique_id->capacity()
            << " byte=" << m_items_unique_id->capacity() * sizeof(Int64);

    // TODO: ability to specify whether or not to compact.
    _compactOnlyItems(false);

    // Assumes compression
    m_infos->prepareForDump();
    m_item_shared_infos->prepareForDump();
    m_need_prepare_dump = true;
  }
  m_item_need_prepare_dump = false;
  if (m_need_prepare_dump) {
    Integer nb_item = m_infos->nbItem();
    // TODO: look into whether it would be better to do this in finishCompactItem()
    _resizeItemVariables(nb_item, true);
    m_internal_variables->m_current_id = m_current_id;
    info(4) << " SET FAMILY ID name=" << name() << " id= " << m_current_id
            << " saveid=" << m_internal_variables->m_current_id();
    ItemInternalList items(m_infos->itemsInternal());
    m_internal_variables->m_items_shared_data_index.resize(nb_item);
    IntegerArrayView items_shared_data_index(m_internal_variables->m_items_shared_data_index);
    info(4) << "ItemFamily::prepareForDump(): " << m_name
            << " count=" << nb_item << " currentid=" << m_current_id;
    // Normally items[i]->localId()==i for all entities because we performed a compaction.
    if (arcaneIsCheck()) {
      for (Integer i = 0; i < nb_item; ++i) {
        ItemInternal* item = items[i];
        if (item->localId() != i)
          ARCANE_FATAL("Incoherence between index ({0}) and localId() ({1})", i, item->localId());
      }
    }
    for (Integer i = 0; i < nb_item; ++i) {
      ItemInternal* item = items[i];
      ItemSharedInfoWithType* isi = m_item_shared_infos->findSharedInfo(item->typeInfo());
      items_shared_data_index[i] = isi->index();
#if 0
#ifdef ARCANE_DEBUG
      //if (itemKind()==IK_Particle){
      info() << "Item: SHARED_INDEX = " << items_shared_data_index[i]
             << " uid = " << item->uniqueId()
             << " lid = " << item->localId()
             << " dataindex = " << item->dataIndex()
             << " flags = " << item->flags();
      //}
#endif
#endif
    }

    // Family linking data
    {
      if (m_parent_family) {
        m_internal_variables->m_parent_family_name = m_parent_family->name();
        m_internal_variables->m_parent_mesh_name = m_parent_family->mesh()->name();
      }
      m_internal_variables->m_parent_family_depth = m_parent_family_depth;
      const Integer child_count = m_child_families.size();
      m_internal_variables->m_child_meshes_name.resize(child_count);
      m_internal_variables->m_child_families_name.resize(child_count);
      for (Integer i = 0; i < child_count; ++i) {
        m_internal_variables->m_child_meshes_name[i] = m_child_families[i]->mesh()->name();
        m_internal_variables->m_child_families_name[i] = m_child_families[i]->name();
      }
    }

    {
      // Determines the number of groups and entities to save.
      // We do not save dynamically generated groups
      Integer nb_group_to_save = 0;
      for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
        const ItemGroup& group = *i;
        if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
          continue;
        debug(Trace::High) << "Save group info name=" << group.name();
        ++nb_group_to_save;
      }
      m_internal_variables->m_groups_name.resize(nb_group_to_save);
      {
        Integer current_group_index = 0;
        for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
          const ItemGroup& group = *i;
          if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
            continue;
          m_internal_variables->m_groups_name[current_group_index] = group.name();
          ++current_group_index;
        }
      }
    }
  }
  // Ensures that the groups are up to date, to ensure that
  // the save will be correct.
  // NOTE: Should this be done here?
  // NOTE: better to use an observer on the group variable?
  _applyCheckNeedUpdateOnGroups();

  m_need_prepare_dump = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
readFromDump()
{
  // TODO: GG: use a flag to indicate that synchronization info needs to be rebuilt
  // but not do it directly in this method.

  Int32 nb_item = 0;
  Int32 dump_version = 0;
  // Indicates if we use the variable containing the entity type for
  // to build the ItemInternal. This is only possible with protections
  // performed since version 3.7 of Arcane. Before that, we must use
  // the m_items_shared_data_index variable.
  bool use_type_variable = false;
  {
    Int32 x = 0;
    auto* p = m_properties;
    if (p->get("dump-version", x))
      dump_version = x;
    if (dump_version >= 0x0307) {
      use_type_variable = true;
      nb_item = p->getInt32("nb-item");
      Int32 cid = p->getInt32("current-change-id");
      Int32 expected_cid = m_internal_variables->m_current_id();
      if (cid != expected_cid)
        ARCANE_FATAL("Bad value for current id mesh={0} id={1} expected={2}",
                     fullName(), cid, expected_cid);
    }
  }

  const bool allow_old_version = true;
  if (!allow_old_version)
    if (dump_version < 0x0307)
      ARCANE_FATAL("Your checkpoint is from a version of Arcane which is too old (mininum version is 3.7)");

  // The part number can change during recovery. Therefore, we must
  // update it. Similarly, if we transition from a sequential mesh to a
  // multi-part mesh during recovery, we must remove isOwn() from the group of all entities.
  const MeshPartInfo& part_info = m_mesh->meshPartInfo();
  m_sub_domain_id = part_info.partRank();
  if (m_infos->allItems().isOwn() && part_info.nbPart() > 1)
    m_infos->allItems().setOwn(false);

  // NOTE: the current implementation assumes that the dataIndex() of
  // entities are consecutive and increasing with the localId() of the entities
  // (i.e., the entity with localId() of 0 also has a dataIndex() of 0,
  // the one with localId() of 1, the next dataIndex() ...)
  // This condition is true if compactReferences() has been called.
  // When this is no longer the case (gap in numbering), we will need to
  // add a data_index variable to the entities.
  IntegerArrayView items_shared_data_index(m_internal_variables->m_items_shared_data_index);
  if (!use_type_variable)
    nb_item = items_shared_data_index.size();

  info(4) << "ItemFamily::readFromDump(): " << fullName()
          << " count=" << nb_item
          << " currentid=" << m_current_id
          << " saveid=" << m_internal_variables->m_current_id()
          << " use_type_variable?=" << use_type_variable
          << " dump_version=" << dump_version;

  if (!use_type_variable) {
    // With older protections, there is no variable for the entity type.
    // We must allocate it here because we use it when calling ItemInternal::setSharedInfo().
    if (nb_item > 0)
      MeshUtils::checkResizeArray(*m_items_type_id, nb_item + 1, false);
    // There is also no offset of 1 for flags, owner and uniqueId.
    // We do this offset here.
    _handleOldCheckpoint();
  }
  _updateItemViews();

  if (m_internal_variables->m_current_id() == m_current_id) {
    debug() << "Family unchanged. Nothing to do.";
    //GG: we still need to recalculate the synchronization info because this family
    // on other sub-domains might have changed and in this case this
    // function will be called. Similarly, the list of groups might have changed
    // and their value also for the calculated groups so we must invalidate them
    _checkComputeSynchronizeInfos(0);
    _readGroups();
    _invalidateComputedGroups();
    return;
  }

  m_current_id = m_internal_variables->m_current_id();
  // IMPORTANT: reset to zero to force resizing of variables
  // upon the next addition of entities.
  m_current_variable_item_size = 0;

  // Family linking data
  {
    IMeshMng* mesh_mng = m_mesh->meshMng();
    if (!m_internal_variables->m_parent_mesh_name().null()) {
      IMesh* parent_mesh = mesh_mng->findMeshHandle(m_internal_variables->m_parent_mesh_name()).mesh();
      m_parent_family = parent_mesh->findItemFamily(m_internal_variables->m_parent_family_name(), true); // true=> fatal if not found
    }
    m_parent_family_depth = m_internal_variables->m_parent_family_depth();
    ARCANE_ASSERT((m_internal_variables->m_child_meshes_name.size() == m_internal_variables->m_child_families_name.size()),
                  ("Incompatible child mesh/family sizes"));
    Integer child_count = m_internal_variables->m_child_families_name.size();
    for (Integer i = 0; i < child_count; ++i) {
      IMesh* child_mesh = mesh_mng->findMeshHandle(m_internal_variables->m_child_meshes_name[i]).mesh();
      IItemFamily* child_family = child_mesh->findItemFamily(m_internal_variables->m_child_families_name[i], true); // true=> fatal if not found
      m_child_families.add(dynamic_cast<ItemFamily*>(child_family));
    }
  }

  m_item_shared_infos->readFromDump();
  m_infos->readFromDump();

  // When reading from dump, the entities are compacted, so the max value of localId()
  // is equal to the number of entities.

  if (use_type_variable) {
    ItemTypeMng* type_mng = mesh()->itemTypeMng();
    for (Integer i = 0; i < nb_item; ++i) {
      ;
      ItemTypeId type_id{ m_common_item_shared_info->m_type_ids[i] };
      ItemSharedInfoWithType* isi = _findSharedInfo(type_mng->typeFromId(type_id));
      Int64 uid = m_items_unique_id_view[i];
      ItemInternal* item = m_infos->allocOne(uid);
      item->_setSharedInfo(isi->sharedInfo(), type_id);
    }
  }
  else {
    // Method used for protections from versions 3.6 and earlier of Arcane.
    auto item_shared_infos = m_item_shared_infos->itemSharedInfos();
    for (Integer i = 0; i < nb_item; ++i) {
      Integer shared_data_index = items_shared_data_index[i];
      ItemSharedInfoWithType* isi = item_shared_infos[shared_data_index];
      Int64 uid = m_items_unique_id_view[i];
      ItemInternal* item = m_infos->allocOne(uid);
      item->_setSharedInfo(isi->sharedInfo(), isi->itemTypeId());
    }
  }

  // Clear the entities from the total group because they will be updated
  // when calling _endUpdate()
  m_infos->allItems().clear();

  // Notifies the source connectivities that we just read from dump.
  for (auto& c : m_source_incremental_item_connectivities)
    c->notifyReadFromDump();

  // Recreation of groups if necessary
  _readGroups();

  // Invalidates the recalculated groups
  _invalidateComputedGroups();

  _endUpdate(false);

  _checkComputeSynchronizeInfos(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_applyCheckNeedUpdateOnGroups()
{
  for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
    ItemGroup group = *i;
    // No need to recalculate the group of global entities
    if (group == m_infos->allItems())
      continue;
    group.internal()->checkNeedUpdate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_invalidateComputedGroups()
{
  // If the group has a parent, it does not have an associated variable and
  // furthermore, it may contain invalid values following a rollback.
  // In this case, we clear it and invalidate it.
  for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
    ItemGroup group = *i;
    if (!group.internal()->parentGroup().null()) {
      group.clear();
      group.invalidate();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Relit les groupes depuis une protection et les recréer si besoin.
 */
void ItemFamily::
_readGroups()
{
  // Recréation des groupes si nécessaire
  VariableArrayString& groups_var = m_internal_variables->m_groups_name;
  debug() << "ItemFamily::readFromDump(): number of group: " << groups_var.size();
  for (Integer i = 0, is = groups_var.size(); i < is; ++i) {
    String name(groups_var[i]);
    debug() << "Readign group again: " << name;
    ItemGroup group = findGroup(name);
    if (group.null())
      createGroup(name);
  }
  // Notifie les groupes qu'ils ont été mis à jour de manière
  // externe. Cela peut être nécessaire pour recalculer automatiquement
  // certaines informations (comme le padding pour la vectorisation)
  for ( ItemGroup& group : m_item_groups) {
    group.incrementTimestamp();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test collectif permettant de savoir s'il faut mettre
 * à jour les infos de synchro.
 *
 * \a changed is 0 if no update, 1 otherwise.
 */
void ItemFamily::
_checkComputeSynchronizeInfos(Int32 changed)
{
  Int32 global_changed = parallelMng()->reduce(Parallel::ReduceMax, changed);
  if (global_changed != 0)
    computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the entities without updating references.
 *
 * If this method is called, you must ensure that
 * compactReference() is called afterward, otherwise the itemsData() array
 * will grow over time.
 */
void ItemFamily::
_compactOnlyItems(bool do_sort)
{
  _compactItems(do_sort);

  // It is necessary to update the groups.
  // TODO verify if this needs to be done all the time
  m_need_prepare_dump = true;

  // Indicates that a compactReference() will also be required
  // during the dump.
  // NOTE: specifying this will also force a recompacting during prepareForDump()
  // and this compaction is unnecessary in this case.
  // TODO: look into how to indicate to prepareForDump() that we only want to
  // perform a compactReference().
  m_item_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the entities.
 */
void ItemFamily::
compactItems(bool do_sort)
{
  _compactOnlyItems(do_sort);

  if (!m_use_legacy_compact_item) {
    // It is necessary to update the groups
    // after a compactReferences().
    _applyCheckNeedUpdateOnGroups();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the entities.
 */
void ItemFamily::
_compactItems(bool do_sort)
{
  IMeshCompactMng* compact_mng = mesh()->_compactMng();
  IMeshCompacter* compacter = compact_mng->beginCompact(this);

  try {
    compacter->setSorted(do_sort);
    compacter->doAllActions();
  }
  catch (...) {
    compact_mng->endCompact();
    throw;
  }
  compact_mng->endCompact();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
beginCompactItems(ItemFamilyCompactInfos& compact_infos)
{
  m_infos->beginCompactItems(compact_infos);

  if (arcaneIsCheck())
    m_infos->checkValid();

  Int32ConstArrayView new_to_old_ids = compact_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = compact_infos.oldToNewLocalIds();

  for (auto& c : m_source_incremental_item_connectivities)
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for (auto& c : m_target_incremental_item_connectivities)
    c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  for (IItemConnectivity* c : m_source_item_connectivities)
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for (IItemConnectivity* c : m_target_item_connectivities)
    c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  if (m_connectivity_mng)
    m_connectivity_mng->notifyLocalIdChanged(this, old_to_new_ids, nbItem());

  // Compacting internal variables associated with the entities
  {
    if (m_parent_family_depth > 0)
      m_internal_variables->m_items_nb_parent.variable()->compact(new_to_old_ids);

    _updateItemViews();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
finishCompactItems(ItemFamilyCompactInfos& compact_infos)
{
  if (arcaneIsCheck())
    m_infos->checkValid();

  m_infos->finishCompactItems(compact_infos);

  for (ItemConnectivitySelector* ics : m_connectivity_selector_list)
    ics->compactConnectivities();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Copies the values of entities numbered @a source into the entities
 * numbered @a destination
 *
 * @param source list of @b source localIds
 * @param destination list of @b destination localIds
 */
void ItemFamily::
copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination)
{
  ARCANE_ASSERT(source.size() == destination.size(),
                ("Can't copy. Source and destination have different size !"));

  if (source.size() != 0) {
    for (IVariable* var : m_used_variables) {
      // (HP) : as seen with Gilles and Stéphane, we do not apply a filter at this level
      // // if the variable is temporary or no restore, we do not copy it
      // if (!(var->property() & (IVariable::PTemporary | IVariable::PNoRestore))) {
      //if (var->itemFamily()==this) {
      var->copyItemsValues(source, destination);
    }
    for (IVariable* var : m_used_shmem_variables) {
      var->copyItemsValues(source, destination);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Copies the average values of entities numbered
 * @a first_source and @a second_source into the entities numbered
 * @a destination
 *
 * @param first_source list of @b localIds of the 1st source
 * @param second_source  list of @b localIds of the 2nd source
 * @param destination  list of @b destination localIds
 */
void ItemFamily::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  ARCANE_ASSERT(first_source.size() == destination.size(),
                ("Can't copy. : first_source and destination have different size !"));
  ARCANE_ASSERT(second_source.size() == destination.size(),
                ("Can't copy : second_source and destination have different size !"));

  if (first_source.size() != 0) {
    for (IVariable* var : m_used_variables) {
      if (!(var->property() & (IVariable::PTemporary | IVariable::PNoRestore))) {
        var->copyItemsMeanValues(first_source, second_source, destination);
      }
    }
    for (IVariable* var : m_used_shmem_variables) {
      if (!(var->property() & (IVariable::PTemporary | IVariable::PNoRestore))) {
        var->copyItemsMeanValues(first_source, second_source, destination);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the variables and groups.
 *
 * \warning: This method must be called during a compaction
 * (between a call to m_infos->beginCompactItems() and m_info->endCompactItems()).
 */
void ItemFamily::
compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos)
{
  Int32ConstArrayView new_to_old_ids = compact_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = compact_infos.oldToNewLocalIds();

  for (IVariable* var : m_used_variables) {
    debug(Trace::High) << "Compact variable " << var->fullName();
    var->compact(new_to_old_ids);
  }
  for (IVariable* var : m_used_shmem_variables) {
    debug(Trace::High) << "Compact shmem variable " << var->fullName();
    var->compact(new_to_old_ids);
  }

  m_variable_synchronizer->changeLocalIds(old_to_new_ids);

  for (ItemGroupList::Enumerator i(m_item_groups); ++i;) {
    ItemGroup group = *i;
    debug(Trace::High) << "Change group Ids: " << group.name();
    group.internal()->changeIds(old_to_new_ids);
    if (group.hasSynchronizer())
      group.synchronizer()->changeLocalIds(old_to_new_ids);
  }

  for (Integer i = 0; i < m_child_families.size(); ++i)
    m_child_families[i]->_compactFromParentFamily(compact_infos);

  info(4) << "End compact family=" << fullName()
          << " max_local_id=" << maxLocalId()
          << " nb_item=" << nbItem();

  // After compaction, variables will be allocated with the number
  // of elements being the number of entities (this is in DynamicMeshKindInfos::finishCompactItems()
  // where maxLocalId() becomes equal to nbItem()).
  m_current_variable_item_size = nbItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compresses the connectivities.
 *
 * \warning: This method must be called during a compaction
 * (between a call to m_infos.beginCompactItems() and m_infos.endCompactItems()).
 */
#ifdef NEED_MERGE
void ItemFamily::
compactConnectivities()
{
  Int32ConstArrayView new_to_old_ids = m_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = m_infos.oldToNewLocalIds();
  for (IItemConnectivity* c : m_source_connectivities)
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for (IItemConnectivity* c : m_target_connectivities)
    c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  if (m_connectivity_mng)
    m_connectivity_mng->notifyLocalIdChanged(this, old_to_new_ids, nbItem());
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_compactFromParentFamily(const ItemFamilyCompactInfos& compact_infos)
{
  debug() << "Compacting child family " << fullName();
  if (m_parent_family_depth > 1)
    throw NotImplementedException(A_FUNCINFO, "Too deep parent family: not yet implemented");
  Int32ConstArrayView old_to_new_lids(compact_infos.oldToNewLocalIds());
  ARCANE_ASSERT((nbItem() == 0 || !old_to_new_lids.empty()), ("Empty oldToNewLocalIds"));
  debug() << "\tfrom parent family " << m_parent_family->name();
  if (this == m_parent_family)
    return; // already self compacted
  ItemInternalArrayView items(itemsInternal());
  for (Integer z = 0, zs = items.size(); z < zs; ++z) {
    ItemInternal* item = items[z];
    Int32 old_parent_lid = item->parentId(0); // depth==1 only !!
    item->setParent(0, old_to_new_lids[old_parent_lid]);
  }
  // If depth>1, it would be better to propagate the compaction by modifying the
  // oldToNewLocalIds of the current sub-mesh families and calling
  // DynamicMesh::_compactItems in cascade (starting from this sub-mesh)
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
internalRemoveItems(Int32ConstArrayView local_ids, bool keep_ghost)
{
  ARCANE_UNUSED(local_ids);
  ARCANE_UNUSED(keep_ghost);
  ARCANE_THROW(NotSupportedException, "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_checkValid()
{
  // Checks that the part number is the same as the mesh's part number
  {
    Int32 part_rank = m_mesh->meshPartInfo().partRank();
    if (m_sub_domain_id != part_rank)
      ARCANE_FATAL("Family {0} Bad value for partRank ({1}) expected={2}",
                   fullName(), m_sub_domain_id, part_rank);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_reserveInfosMemory(Integer memory)
{
  ARCANE_UNUSED(memory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_resizeInfos(Integer new_size)
{
  ARCANE_UNUSED(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
_allocMany(Integer memory)
{
  ARCANE_UNUSED(memory);
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoWithType* ItemFamily::
_findSharedInfo(ItemTypeInfo* type)
{
  return m_item_shared_infos->findSharedInfo(type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateSharedInfo()
{
  //TODO: Check if this is still useful
  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_allocateInfos(ItemInternal* item, Int64 uid, ItemTypeInfo* type)
{
  ItemSharedInfoWithType* isi = _findSharedInfo(type);
  _allocateInfos(item, uid, isi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_resizeItemVariables(Int32 new_size, bool force_resize)
{
  bool is_resize = MeshUtils::checkResizeArray(*m_items_unique_id, new_size + 1, force_resize);
  is_resize |= MeshUtils::checkResizeArray(*m_items_owner, new_size + 1, force_resize);
  is_resize |= MeshUtils::checkResizeArray(*m_items_flags, new_size + 1, force_resize);
  is_resize |= MeshUtils::checkResizeArray(*m_items_type_id, new_size + 1, force_resize);
  if (m_parent_family_depth > 0)
    is_resize |= MeshUtils::checkResizeArray(*m_items_nb_parent, new_size, force_resize);
  if (is_resize)
    _updateItemViews();

  // Positions the values for the null entity.
  // NOTE: check if this should be done at initialization.
  (*m_items_unique_id)[0] = NULL_ITEM_UNIQUE_ID;
  (*m_items_flags)[0] = 0;
  (*m_items_owner)[0] = A_NULL_RANK;
  (*m_items_type_id)[0] = IT_NullType;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_allocateInfos(ItemInternal* item, Int64 uid, ItemSharedInfoWithType* isi)
{
  // TODO: do simultaneously with the realloc of the uniqueId() variable
  //  the realloc of the m_source_incremental_item_connectivities.
  Int32 local_id = item->localId();
  _resizeItemVariables(local_id + 1, false);

  // TODO: check if still useful because ItemInternal::reinitialize() must do it
  //(*m_items_unique_id)[local_id] = uid;

  ItemTypeId iti = isi->itemTypeId();
  item->_setSharedInfo(isi->sharedInfo(), iti);

  item->reinitialize(uid, m_default_sub_domain_owner, m_sub_domain_id);
  ++m_nb_allocate_info;
  // Notify the incremental connectivities that we added an item to the source
  for (auto& c : m_source_incremental_item_connectivities)
    c->notifySourceItemAdded(ItemLocalId(local_id));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_preAllocate(Int32 nb_item, bool pre_alloc_connectivity)
{
  if (nb_item > 1000)
    m_infos->itemsMap().resize(nb_item, true);
  _resizeItemVariables(nb_item, false);
  for (auto& c : m_source_incremental_item_connectivities)
    c->reserveMemoryForNbSourceItems(nb_item, pre_alloc_connectivity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_notifyDataIndexChanged()
{
  _updateItemViews();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
getCommunicatingSubDomains(Int32Array& sub_domains) const
{
  Int32ConstArrayView ranks = m_variable_synchronizer->communicatingRanks();
  Integer s = ranks.size();
  sub_domains.resize(s);
  sub_domains.copy(ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
computeSynchronizeInfos()
{
  // Only calculate synchronization info if we are in parallel
  // and the number of parts equals the number of ranks of the parallelMng(),
  // which is not the case during recovery after a change in the number of
  // sub-domains.
  if (m_is_parallel && m_mesh->meshPartInfo().nbPart() == parallelMng()->commSize()) {
    m_variable_synchronizer->compute();
    _updateItemsSharedFlag();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO move this method outside of ItemFamily
void ItemFamily::
reduceFromGhostItems(IVariable* v, IDataOperation* operation)
{
  if (!v)
    return;
  if (v->itemFamily() != this && v->itemGroup().itemFamily() != this)
    throw ArgumentException(A_FUNCINFO, "Variable not in this family");
  Parallel::GhostItemsVariableParallelOperation op(this);
  op.setItemFamily(this);
  op.addVariable(v);
  op.applyOperation(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO move this method outside of ItemFamily
void ItemFamily::
reduceFromGhostItems(IVariable* v, Parallel::eReduceType reduction)
{
  ScopedPtrT<IDataOperation> operation;
  operation = v->dataFactoryMng()->createDataOperation(reduction);
  reduceFromGhostItems(v, operation.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ArrayView* ItemFamily::
uniqueIds()
{
  return &m_items_unique_id_view;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* ItemFamily::
findVariable(const String& var_name, bool throw_exception)
{
  IVariableMng* vm = subDomain()->variableMng();
  StringBuilder vname = mesh()->name();
  vname += "_";
  vname += name();
  vname += "_";
  vname += var_name;
  IVariable* var = vm->findVariableFullyQualified(vname.toString());
  if (!var && throw_exception) {
    ARCANE_FATAL("No variable named '{0}' in family '{1}'", var_name, name());
  }
  return var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
clearItems()
{
  m_infos->clear();

  endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
checkUniqueIds(Int64ConstArrayView unique_ids)
{
  Integer nb_item = unique_ids.size();
  Int64UniqueArray all_unique_ids;
  IParallelMng* pm = m_mesh->parallelMng();
  pm->allGatherVariable(unique_ids, all_unique_ids);
  HashTableMapT<Int64, Integer> items_map(nb_item * 2, true);
  info() << "ItemFamily::checkUniqueIds name=" << name() << " n=" << nb_item
         << " total=" << all_unique_ids.size();
  for (Integer i = 0; i < nb_item; ++i)
    items_map.add(unique_ids[i], 0);
  for (Integer i = 0, is = all_unique_ids.size(); i < is; ++i) {
    HashTableMapT<Int64, Integer>::Data* data = items_map.lookup(all_unique_ids[i]);
    if (data)
      ++data->value();
  }
  for (Integer i = 0; i < nb_item; ++i) {
    Integer nb_ref = items_map[unique_ids[i]];
    if (nb_ref != 1) {
      fatal() << "Duplicate unique_id=" << unique_ids[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup ItemFamily::
findAdjencyItems(const ItemGroup& group, const ItemGroup& sub_group,
                 eItemKind link_kind, Integer layer)
{
  return findAdjacencyItems(group, sub_group, link_kind, layer);
}

ItemPairGroup ItemFamily::
findAdjacencyItems(const ItemGroup& group, const ItemGroup& sub_group,
                   eItemKind link_kind, Integer layer)
{
  AdjacencyInfo at(group, sub_group, link_kind, layer);
  auto i = m_adjacency_groups.find(at);

  if (i == m_adjacency_groups.end()) {
    debug() << "** BUILD ADJENCY_ITEMS : " << group.name() << " x "
            << sub_group.name() << " link=" << link_kind << " nblayer=" << layer;
    ItemPairGroup v(new ItemPairGroupImpl(group,sub_group));
    mesh()->utilities()->computeAdjacency(v, link_kind, layer);
    m_adjacency_groups.insert(std::make_pair(at, v));
    return v;
  }
  debug() << "** FOUND KNOWN ADJENCY_ITEMS! : " << group.name() << " x "
          << sub_group.name() << " link=" << link_kind << " nblayer=" << layer;
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemConnectivityInfo* ItemFamily::
localConnectivityInfos() const
{
  return m_local_connectivity_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemConnectivityInfo* ItemFamily::
globalConnectivityInfos() const
{
  return m_global_connectivity_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setHasUniqueIdMap(bool v)
{
  ARCANE_UNUSED(v);
  throw NotSupportedException(A_FUNCINFO,"this kind of family doesn't support this function");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemFamily::
hasUniqueIdMap() const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemFamily::
view(Int32ConstArrayView local_ids)
{
  return ItemVectorView(itemInfoListView(),local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemFamily::
view()
{
  return allItems().view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_addVariable(IVariable* var)
{
  //info() << "Add var=" << var->fullName() << " to family=" << name();
  if (var->itemFamily()!=this)
    ARCANE_FATAL("Can not add a variable to a different family");

  if (var->property() & IVariable::PInShMem)
    m_used_shmem_variables.insert(var);
  else
    m_used_variables.insert(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_removeVariable(IVariable* var)
{
  //info() << "Remove var=" << var->fullName() << " to family=" << name();
  if (var->property() & IVariable::PInShMem)
    m_used_shmem_variables.erase(var);
  else
    m_used_variables.erase(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
usedVariables(VariableCollection collection)
{
  for( IVariable* var : m_used_variables ){
    collection.add(var);
  }
  for (IVariable* var : m_used_shmem_variables) {
    collection.add(var);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* ItemFamily::
allItemsSynchronizer()
{
  return m_variable_synchronizer.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setItemSortFunction(IItemInternalSortFunction* sort_function)
{
  if (m_item_sort_function==sort_function)
    return;
  delete m_item_sort_function;
  m_item_sort_function = sort_function;
  if (!m_item_sort_function)
    m_item_sort_function = _defaultItemSortFunction();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemInternalSortFunction* ItemFamily::
itemSortFunction() const
{
  return m_item_sort_function;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
synchronize(VariableCollection variables)
{
  m_variable_synchronizer->synchronize(variables);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
synchronize(VariableCollection variables, Int32ConstArrayView local_ids)
{
  m_variable_synchronizer->synchronize(variables, local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items, Int32ConstArrayView owners)
{
  ARCANE_UNUSED(unique_ids);
  ARCANE_UNUSED(items);
  ARCANE_UNUSED(owners);
  ARCANE_THROW(NotImplementedException,"this kind of family doesn't support this operation yet. Only DoF at present.");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeItems2(ItemDataList& item_data_list)
{
  if (!m_mesh->itemFamilyNetwork())
    ARCANE_FATAL("Family name='{0}': IMesh::itemFamilyNetwork() is null",name());

  ItemData& item_data = item_data_list[itemKind()];
  // 1-Prepare : Get child connectivities and families
  SharedArray<IIncrementalItemConnectivity*> child_connectivities = m_mesh->itemFamilyNetwork()->getChildDependencies(this); // TODO : change return type ? (List to Array ?)
  UniqueArray<IItemFamily*> child_families;
  UniqueArray<IIncrementalItemConnectivity*> child_families_to_current_family;
  UniqueArray<ItemScalarProperty<bool>> child_families_has_extra_parent_properties; // To indicate wheter child items have other parents.
  Integer index = 0;
  for (auto child_connectivity : child_connectivities) {
    child_families.add(child_connectivity->targetFamily());
    child_families_to_current_family.add(m_mesh->itemFamilyNetwork()->getConnectivity(child_families.back(),this,connectivityName(child_families.back(),this)));
    if (child_families_to_current_family.back() == nullptr) fatal() << "removeItems2 needs reverse connectivity. Missing Connectivity " << connectivityName(child_families.back(),this);
    child_families_has_extra_parent_properties.add(ItemScalarProperty<bool>());
    child_families_has_extra_parent_properties.back().resize(child_families.back(),false);
    for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentDependencies(child_families.back())) { // exclure parent actuel
        if (parent_connectivity == child_connectivity) continue;
        ItemVector connected_items;
        _getConnectedItems(parent_connectivity,connected_items);
        _fillHasExtraParentProperty(child_families_has_extra_parent_properties[index],connected_items);
    }
    index++;
  }
  // 2-Propagates item removal to child families
  Int64ArrayView removed_item_lids = item_data.itemInfos().view(); // Todo change ItemData to store removed item lids in Int32
  for (auto removed_item_lid_int64 : removed_item_lids) {
    Int32 removed_item_lid = CheckedConvert::toInt32(removed_item_lid_int64);
    index = 0;
    for (auto child_connectivity : child_connectivities) {
      ConnectivityItemVector child_con_accessor(child_connectivity);
      ENUMERATE_ITEM(connected_item, child_con_accessor.connectedItems(ItemLocalId(removed_item_lid))) {
        if (!this->itemsInternal()[removed_item_lid]->isDetached()) {// test necessary when doing removeDetached (the relations are already deleted).
          child_families_to_current_family[index]->removeConnectedItem(ItemLocalId(connected_item),ItemLocalId(removed_item_lid));
        }
        // Check if connected item is to remove
        if (! child_families_has_extra_parent_properties[index][connected_item] && child_families_to_current_family[index]->nbConnectedItem(ItemLocalId(connected_item)) == 0) {
          item_data_list[child_connectivity->targetFamily()->itemKind()].itemInfos().add((Int64) connected_item.localId());
        }
      }
      index++;
    }
  }
  // => merge this loop with previous one ?
  // 3-1 Remove relations for child relations
  for (auto removed_item_lid_int64 : removed_item_lids) {
    Int32 removed_item_lid = CheckedConvert::toInt32(removed_item_lid_int64);
    for (auto child_relation : m_mesh->itemFamilyNetwork()->getChildRelations(this)) {
      ConnectivityItemVector connectivity_accessor(child_relation);
      ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(removed_item_lid))) {
        child_relation->removeConnectedItem(ItemLocalId(removed_item_lid),ItemLocalId(connected_item));
      }
    }
  }
  // 3-2 Remove relations for parent relations
  ItemScalarProperty<bool> is_removed_item;
  is_removed_item.resize(this,false);
  for (auto removed_item_lid_int64 : removed_item_lids) {
    Int32 removed_item_lid = CheckedConvert::toInt32(removed_item_lid_int64);
    is_removed_item[*(this->itemsInternal()[removed_item_lid])] = true;
  }
  for (auto parent_relation : m_mesh->itemFamilyNetwork()->getParentRelations(this)) {
    for (auto source_item : parent_relation->sourceFamily()->itemsInternal()) {
        if (source_item->isSuppressed()) continue;
      ConnectivityItemVector connectivity_accessor(parent_relation);
      ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(source_item))) {
        if (is_removed_item[connected_item])
          parent_relation->removeConnectedItem(ItemLocalId(source_item),connected_item);
      }
    }
  }
  // 4-Remove items. Child items will be removed by an automatic call of removeItems2 on their family...
  for (auto removed_item_lid_int64 : removed_item_lids) {
    Int32 removed_item_lid = CheckedConvert::toInt32(removed_item_lid_int64);
    ItemInternal* removed_item = m_infos->itemInternal(removed_item_lid);
    if (removed_item->isDetached()) {
      m_infos->removeDetachedOne(removed_item);
    }
    else {
      m_infos->removeOne(removed_item);
    }
  }
  this->endUpdate();// endUpdate is needed since we then go deeper in the dependency graph and will need to enumerate this changed family.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_getConnectedItems(IIncrementalItemConnectivity* parent_connectivity,ItemVector& target_family_connected_items)
{
  ConnectivityItemVector connectivity_accessor(parent_connectivity);
  for(auto source_item : parent_connectivity->sourceFamily()->itemsInternal()) {
    if (source_item->isSuppressed()) continue;
    target_family_connected_items.add(connectivity_accessor.connectedItems(ItemLocalId(source_item)).localIds());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_fillHasExtraParentProperty(ItemScalarProperty<bool>& child_families_has_extra_parent,ItemVectorView connected_items)
{
  ENUMERATE_ITEM(connected_item, connected_items) {
    child_families_has_extra_parent[connected_item] = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_detachCells2(Int32ConstArrayView local_ids)
{
  //- Only cells are detached, i.e., no parent dependencies are to be found
  ARCANE_ASSERT((m_mesh->itemFamilyNetwork()->getParentDependencies(this).empty()),("Only cells are detached, no parent dependencies are to be found."))
  // Remove all parent and child relations. Keep child dependencies => used when removing detached cells. No parent dependencies
  // 1 Remove relations for child relations
  for (auto removed_item_lid : local_ids) {
      for (auto child_relation : m_mesh->itemFamilyNetwork()->getChildRelations(this)) {
        ConnectivityItemVector connectivity_accessor(child_relation);
        ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(removed_item_lid))) {
          child_relation->removeConnectedItem(ItemLocalId(removed_item_lid),ItemLocalId(connected_item));
        }
      }
  }
  // 2 Remove relations for parent relations
  ItemScalarProperty<bool> is_detached_item;
  is_detached_item.resize(this,false);
  for (auto detached_item_lid : local_ids) {
      is_detached_item[*(this->itemsInternal()[detached_item_lid])] = true;
  }
  for (auto parent_relation : m_mesh->itemFamilyNetwork()->getParentRelations(this)) {
    for(auto source_item : parent_relation->sourceFamily()->itemsInternal()) {
      if (source_item->isSuppressed()) continue;
      ConnectivityItemVector connectivity_accessor(parent_relation);
      ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(source_item))) {
        if (is_detached_item[connected_item]) {
          parent_relation->removeConnectedItem(ItemLocalId(source_item),connected_item);
        }
      }
    }
  }
  // 4-Detach items.
  for (auto detached_item_lid : local_ids) {
    m_infos->detachOne(m_infos->itemInternal(detached_item_lid)); // when family/mesh endUpdate is done? needed?
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeNeedRemoveMarkedItems()
{
  ItemInternalMap& item_map = itemsMap();
  if (item_map.count()==0)
    return;

  if (!m_mesh->itemFamilyNetwork())
    ARCANE_FATAL("Family name='{0}': IMesh::itemFamilyNetwork() is null",name());
  if (!IItemFamilyNetwork::plug_serializer)
    ARCANE_FATAL("family name='{0}': removeNeedMarkedItems() cannot be called if ItemFamilyNetwork is unplugged.",name());

  UniqueArray<ItemInternal*> items_to_remove;
  UniqueArray<Int32> items_to_remove_lids;
  items_to_remove.reserve(1000);
  items_to_remove_lids.reserve(1000);

  item_map.eachItem([&](impl::ItemBase item) {
    Integer f = item.flags();
    if (f & ItemFlags::II_NeedRemove){
      f &= ~ItemFlags::II_NeedRemove & ItemFlags::II_Suppressed;
      item.toMutable().setFlags(f);
      items_to_remove.add(item._itemInternal());
      items_to_remove_lids.add(item.localId());
    }
  });
  info() << "Number of " << itemKind() << " of family "<< name()<<" to remove: " << items_to_remove.size();
  if (items_to_remove.size() == 0)
    return;

  // Update connectivities => remove all con pointing on the removed items
  // TODO the nearly same procedure is done in _detachCells2: mutualize in a method (watch out the connectivities used are parentConnectivities or parentRelations...)
  ItemScalarProperty<bool> is_removed_item;
  is_removed_item.resize(this,false);
  for (auto removed_item: items_to_remove) {
    is_removed_item[*removed_item] = true;
  }
  //for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentConnectivities(this)) {
  for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentRelations(this)) {
    for(auto source_item : parent_connectivity->sourceFamily()->itemsInternal()) {
      if (source_item->isSuppressed()) continue;
      ConnectivityItemVector connectivity_accessor(parent_connectivity);
      ENUMERATE_ITEM(connected_item, connectivity_accessor.connectedItems(ItemLocalId(source_item))) {
        if (is_removed_item[connected_item]) {
          parent_connectivity->removeConnectedItem(ItemLocalId(source_item),connected_item);
        }
      }
    }
  }
  // Remove items
  m_infos->removeMany(items_to_remove_lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CompareUniqueIdWithSuppression
{
 public:
  bool operator()(const ItemInternal* item1,const ItemInternal* item2) const
  {
    // Destroyed entities must be placed at the end of the list.
    //cout << "Compare: " << item1->uniqueId() << " " << item2->uniqueId() << '\n';
    bool s1 = item1->isSuppressed();
    bool s2 = item2->isSuppressed();
    if (s1 && !s2)
      return false;
    if (!s1 && s2)
      return true;
    return item1->uniqueId() < item2->uniqueId();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemInternalSortFunction* ItemFamily::
_defaultItemSortFunction()
{
  return new ItemInternalSortFunction<CompareUniqueIdWithSuppression>("ArcaneUniqueId");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setPolicyMng(IItemFamilyPolicyMng* policy_mng)
{
  if (m_policy_mng==policy_mng)
    return;
  if (m_policy_mng)
    ARCANE_FATAL("PolicyMng already set");
  m_policy_mng = policy_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addSourceConnectivity(IItemConnectivity* connectivity)
{
  m_source_item_connectivities.insert(connectivity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addTargetConnectivity(IItemConnectivity* connectivity)
{
  m_target_item_connectivities.insert(connectivity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeSourceConnectivity(IItemConnectivity* connectivity)
{
  m_source_item_connectivities.erase(m_source_item_connectivities.find(connectivity));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeTargetConnectivity(IItemConnectivity* connectivity)
{
  m_target_item_connectivities.erase(m_target_item_connectivities.find(connectivity));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setConnectivityMng(IItemConnectivityMng* connectivity_mng)
{
  ARCANE_ASSERT((m_connectivity_mng == NULL || m_connectivity_mng== connectivity_mng),
                ("Connectivity Manager must be unique") )
  m_connectivity_mng = connectivity_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObservableView<const ItemFamilyItemListChangedEventArgs&> ItemFamily::
itemListChangedEvent()
{
  return m_infos->itemListChangedEvent();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
experimentalChangeUniqueId(ItemLocalId local_id,ItemUniqueId unique_id)
{
  ItemInternal* iitem = _itemInternal(local_id);
  Int64 old_uid = iitem->uniqueId();
  if (old_uid==unique_id)
    return;
  //MutableItemBase item_base(local_id,m_common_item_shared_info);
  iitem->setUniqueId(unique_id);

  if (m_infos->hasUniqueIdMap()){
    ItemInternalMap& item_map = itemsMap();
    item_map.remove(old_uid);
    item_map.add(unique_id,iitem);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_addSourceConnectivity(IIncrementalItemSourceConnectivity* c)
{
  m_source_incremental_item_connectivities.add(c->toSourceReference());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_addTargetConnectivity(IIncrementalItemTargetConnectivity* c)
{
  m_target_incremental_item_connectivities.add(c->toTargetReference());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_checkValidConnectivity()
{
  {
    // Checks that there are no null entities.
    ENUMERATE_ITEM(i,allItems()){
      Item item = *i;
      if (item.null())
        ARCANE_FATAL("family={0}: local item lid={1} is null",fullName(),item);
    }
  }
  {
    // Checks the consistency of the internal part
    ENUMERATE_ITEM(i,allItems()){
      Item item = *i;
      Item i1 = item;
      Item i2 = m_common_item_shared_info->m_items_internal[item.localId()];
      if (i1!=i2)
        ARCANE_FATAL("family={0}: incoherent item internal lid={1} i1={2} i2={3}",
                     fullName(),item.localId(),ItemPrinter(i1),ItemPrinter(i2));
    }
  }
  constexpr Int32 MAX_KIND = ItemInternalConnectivityList::MAX_ITEM_KIND;
  std::array<Int32,MAX_KIND> computed_max;
  computed_max.fill(0);

  for( Integer i=0; i<MAX_KIND; ++i ){
    eItemKind target_kind = static_cast<eItemKind>(i);
    IndexedItemConnectivityViewBase con_view{m_item_connectivity_list.containerView(i),itemKind(),target_kind};
    Int32 stored_max_nb = m_item_connectivity_list.maxNbConnectedItem(i);
    const Int32 con_nb_item_size = con_view.nbSourceItem();

    info(4) << "Family name=" << fullName() << " I=" << i << " nb_item_size=" << con_nb_item_size;

    Int32 max_nb = 0;
    if (con_nb_item_size!=0){
      // It is necessary to iterate over all entities and not over \a con_nb_item
      // because some values may not be valid if there are
      // gaps in the numbering.
      ENUMERATE_ITEM(i,allItems()){
        Int32 x = con_view.nbItem(i);
        if (x>max_nb)
          max_nb = x;
      }
      if (stored_max_nb<max_nb)
        ARCANE_FATAL("Bad value for max connected item family={0} kind={1} stored={2} computed={3}",
                     name(),i,stored_max_nb,max_nb);
      computed_max[i] = max_nb;
    }
  }
  // Checks that the value returned by m_local_connectivity_info
  // is at least greater than 'computed_max'
  {
    std::array<Int32,MAX_KIND> stored_max;
    stored_max.fill(0);
    auto* ci = m_local_connectivity_info;
    stored_max[ItemInternalConnectivityList::NODE_IDX] = ci->maxNodePerItem();
    stored_max[ItemInternalConnectivityList::EDGE_IDX] = ci->maxEdgePerItem();
    stored_max[ItemInternalConnectivityList::FACE_IDX] = ci->maxFacePerItem();
    stored_max[ItemInternalConnectivityList::CELL_IDX] = ci->maxCellPerItem();
    // For the following two, there is no equivalent in 'ItemConnectivityInfo' so
    // we put the calculated values to avoid generating an error.
    stored_max[ItemInternalConnectivityList::HPARENT_IDX] = computed_max[ItemInternalConnectivityList::HPARENT_IDX];
    stored_max[ItemInternalConnectivityList::HCHILD_IDX] = computed_max[ItemInternalConnectivityList::HCHILD_IDX];
    for( Integer i=0; i<MAX_KIND; ++i )
      if (stored_max[i]<computed_max[i])
        ARCANE_FATAL("Bad value for local_connectivity_info family={0} kind={1} stored={2} computed={3}",
                     name(),i,stored_max[i],computed_max[i]);
  }
  for( auto  ics : m_connectivity_selector_list )
    ics->checkValidConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_addConnectivitySelector(ItemConnectivitySelector* selector)
{
  m_connectivity_selector_list.add(selector);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_buildConnectivitySelectors()
{
  m_connectivity_selector_list_by_item_kind.clear();
  m_connectivity_selector_list_by_item_kind.resize(ItemInternalConnectivityList::MAX_ITEM_KIND);
  m_connectivity_selector_list_by_item_kind.fill(nullptr);

  for( ItemConnectivitySelector* ics : m_connectivity_selector_list ){
    ics->build();
    Int32 i = ics->itemConnectivityIndex();
    if (i>=0){
      if (m_connectivity_selector_list_by_item_kind[i])
        ARCANE_FATAL("Can not have two connectivity selector for same item kind");
      m_connectivity_selector_list_by_item_kind[i] = ics;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_setTopologyModifier(IItemFamilyTopologyModifier* tm)
{
  delete m_topology_modifier;
  m_topology_modifier = tm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positions Item::isShared() info for the entities of the
 * family.
 *
 * This method is only valid if the connectivity info of
 * m_variable_synchronizer is up to date.
 */
void ItemFamily::
_updateItemsSharedFlag()
{
  ItemInternalList items(_itemsInternal());
  for( Integer i=0, n=items.size(); i<n; ++i )
    items[i]->removeFlags(ItemFlags::II_Shared);
  Int32ConstArrayView comm_ranks = m_variable_synchronizer->communicatingRanks();
  Integer nb_rank = comm_ranks.size();
  // Iterates through the synchronizer's sharedItems() and sets the flag
  // II_Shared for the entities in the list.
  for( Integer i=0; i<nb_rank; ++i ){
    Int32ConstArrayView shared_ids = m_variable_synchronizer->sharedItems(i);
    for( auto id : shared_ids )
      items[id]->addFlags(ItemFlags::II_Shared);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_computeConnectivityInfo(ItemConnectivityInfo* ici)
{
  ici->fill(itemInternalConnectivityList());
  info(5) << "COMPUTE CONNECTIVITY INFO family=" << name() << " v=" << ici
          << " node=" << ici->maxNodePerItem() << " face=" << ici->maxFacePerItem()
          << " edge=" << ici->maxEdgePerItem() << " cell=" << ici->maxCellPerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_handleOldCheckpoint()
{
  // No need to manage 'm_items_type_id' because it is not present in
  // the old protections.
  _offsetArrayByOne(m_items_unique_id);
  _offsetArrayByOne(m_items_flags);
  _offsetArrayByOne(m_items_owner);
  (*m_items_unique_id)[0] = NULL_ITEM_UNIQUE_ID;
  (*m_items_flags)[0] = 0;
  (*m_items_owner)[0] = A_NULL_RANK;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

// Returns a view that starts on the second element of the array.
template<typename DataType>
ArrayView<DataType> _getView(Array<DataType>* v)
{
  Int32 n = v->size();
  return v->subView(1,n-1);
 }

}

void ItemFamily::
_updateItemViews()
{
  m_common_item_shared_info->m_unique_ids = _getView(m_items_unique_id);
  m_common_item_shared_info->m_flags = _getView(m_items_flags);
  m_common_item_shared_info->m_type_ids = _getView(m_items_type_id);
  m_common_item_shared_info->m_owners = _getView(m_items_owner);
  m_common_item_shared_info->m_parent_item_ids = m_items_nb_parent->view();

  m_items_unique_id_view = _getView(m_items_unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_addOnSizeChangedObservable(VariableRef& var_ref)
{
  m_observers.addObserver(this,&ItemFamily::_updateItemViews,
                          var_ref.variable()->onSizeChangedObservable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyInternal* ItemFamily::
_internalApi()
{
  return m_internal_api;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
