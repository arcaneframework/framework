﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamily.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Infos de maillage pour un genre d'entité donnée.                          */
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

#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/VariableTypes.h"
#include "arcane/IVariableMng.h"
#include "arcane/IVariable.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IApplication.h"
#include "arcane/IDataFactoryMng.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/ItemPairGroupImpl.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/ItemInternalSortFunction.h"
#include "arcane/Properties.h"
#include "arcane/ItemFamilyCompactInfos.h"
#include "arcane/IMeshMng.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/IMeshCompactMng.h"
#include "arcane/MeshPartInfo.h"
#include "arcane/ParallelMngUtils.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/datatype/IDataOperation.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemSharedInfoList.h"
#include "arcane/mesh/ItemConnectivityInfo.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"

#include "arcane/parallel/GhostItemsVariableParallelOperation.h"
#include "arcane/parallel/IStat.h"

#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/IItemConnectivityMng.h"
#include "arcane/IItemFamilyPolicyMng.h"

#include "arcane/ItemPrinter.h"
#include "arcane/ConnectivityItemVector.h"

#include "arcane/mesh/ItemProperty.h"
#include "arcane/mesh/ItemData.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily::Variables
{
 public:
  Variables(IMesh* mesh,
            const String& family_name,
            eItemKind item_kind,
            const String& shared_data_name,
            const String& data_name,
            const String& unique_ids_name,
            const String& groups_name,
            const String& current_id_name,
            const String& new_owner_name,
            const String& parent_mesh_name,
            const String& parent_family_name,
            const String& parent_family_depth_name,
            const String& child_meshes_name,
            const String& child_families_name)
  : m_items_shared_data_index(VariableBuildInfo(mesh,shared_data_name,IVariable::PPrivate))
    , m_items_data(VariableBuildInfo(mesh,data_name,IVariable::PPrivate))
    , m_items_unique_id(VariableBuildInfo(mesh,unique_ids_name,IVariable::PPrivate))
    , m_groups_name(VariableBuildInfo(mesh,groups_name))
    , m_current_id(VariableBuildInfo(mesh,current_id_name))
    , m_items_new_owner(VariableBuildInfo(mesh,new_owner_name,family_name,
                                          IVariable::PSubDomainDepend|IVariable::PExecutionDepend),item_kind)
    , m_parent_mesh_name(VariableBuildInfo(mesh,parent_mesh_name,IVariable::PPrivate))
    , m_parent_family_name(VariableBuildInfo(mesh,parent_family_name,IVariable::PPrivate))
    , m_parent_family_depth(VariableBuildInfo(mesh,parent_family_depth_name,IVariable::PPrivate))
    , m_child_meshes_name(VariableBuildInfo(mesh,child_meshes_name,IVariable::PPrivate))
    , m_child_families_name(VariableBuildInfo(mesh,child_families_name,IVariable::PPrivate))
    {}
 public:
  void setUsed()
  {
    m_items_new_owner.setUsed(true);
  }
 public:
  VariableArrayInteger m_items_shared_data_index;
  VariableArrayInt32 m_items_data;
  //! Contient les uniqueIds() des entités de cette famille
  VariableArrayInt64 m_items_unique_id;
  VariableArrayString m_groups_name;
  VariableScalarInteger m_current_id;
  /*! \brief Contient le sous-domaine propriétaire de l'entité.
   * Cette variable est redondante avec le champ owner() de ItemInternal
   * et n'a une valeur différente qu'au moment où des entités changent
   * de propriétaire. Par conséquent, il ne devrait pas être nécessaire de l'allouer
   * en séquentiel.
   * \todo De même, on peut retrouver sa valeur lors d'une reprise
   * et il faudra la marquer IVariable::PNoDump.
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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamily::
ItemFamily(IMesh* mesh,eItemKind ik,const String& name)
: TraceAccessor(mesh->traceMng())
, m_name(name)
, m_mesh(mesh)
, m_sub_domain(mesh->subDomain())
, m_parent_family(nullptr)
, m_parent_family_depth(0)
, m_infos(mesh,ik,name)
, m_need_prepare_dump(true)
, m_item_internal_list(mesh->meshItemInternalList())
, m_item_shared_infos(new ItemSharedInfoList(this))
, m_current_variable_item_size(0)
, m_item_sort_function(nullptr)
, m_local_connectivity_info(nullptr)
, m_global_connectivity_info(nullptr)
, m_properties(new Properties(*mesh->properties(),name))
, m_connectivity_mng(nullptr)
, m_policy_mng(nullptr)
, m_items_data(nullptr)
, m_items_unique_id(nullptr)
, m_internal_variables(nullptr)
, m_default_sub_domain_owner(NULL_SUB_DOMAIN_ID)
, m_sub_domain_id(mesh->meshPartInfo().partRank())
, m_is_parallel(false)
, m_current_id(0)
, m_item_need_prepare_dump(false)
, m_nb_allocate_info(0)
, m_topology_modifier(nullptr)
{
  m_item_connectivity_list.m_items = mesh->meshItemInternalList();
  m_infos.setItemFamily(this);
  m_connectivity_selector_list_by_item_kind.resize(ItemInternalConnectivityList::MAX_ITEM_KIND);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamily::
~ItemFamily()
{
  info(4) << "Family name=" << m_full_name
          << " nb_access_v2=" << m_item_connectivity_list.nbAccess()
          << " nb_access_all_v2=" << m_item_connectivity_list.nbAccessAll();

  delete m_topology_modifier;
  delete m_policy_mng;
  delete m_properties;
  delete m_local_connectivity_info;
  delete m_global_connectivity_info;
  delete m_item_sort_function;
  delete m_internal_variables;
  delete m_item_shared_infos;

  for( ItemConnectivitySelector* ics : m_connectivity_selector_list )
    delete ics;

  for( IIncrementalItemConnectivity* c : m_source_incremental_item_connectivities )
    delete c;

  for( Integer i=0; i<ItemInternalConnectivityList::MAX_ITEM_KIND; ++i ){
    delete m_item_connectivity_list.m_indexes_array[i];
    delete m_item_connectivity_list.m_nb_item_array[i];
  }
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
  // Créé le modificateur de topologie si cela n'a pas encore été fait par
  // la classe dérivée.
  if (!m_topology_modifier)
    m_topology_modifier = new AbstractItemFamilyTopologyModifier(this);

  m_full_name = m_mesh->name() + "_" + m_name;

  m_local_connectivity_info = new ItemConnectivityInfo();
  m_global_connectivity_info = new ItemConnectivityInfo();

  IParallelMng* pm = m_mesh->parallelMng();
  if (!pm->isParallel()){
    m_default_sub_domain_owner = 0;
    m_is_parallel = false;
  }
  else
    m_is_parallel = true;

  // D'abord initialiser les infos car cela créé les groupes d'entités
  // et c'est indispensable avant de créer des variables dessus.
  m_infos.build();

  // Construit l'instance qui contiendra les variables
  // NOTE: si on change les noms ici, il faut aussi les changer dans MeshStats.cc
  // sinon les statistiques ne seront pas fiables.
  {
    String var_data_name(_variableName("FamilyItemsData"));
    String var_unique_ids_name(_variableName("FamilyUniqueIds"));
    String var_count_name(_variableName("FamilyItemsShared"));
    String var_groups_name(_variableName("FamilyGroupsName"));
    String var_current_id_name(_variableName("FamilyCurrentId"));
    String var_new_owner_name(_variableName("FamilyNewOwnerName"));
    String var_parent_mesh_name(_variableName("ParentMeshName"));
    String var_parent_family_name(_variableName("ParentFamilyName"));
    String var_parent_family_depth_name(_variableName("ParentFamilyDepthName"));
    String var_child_meshes_name(_variableName("ChildMeshesName"));
    String var_child_families_name(_variableName("ChildFamiliesName"));
    m_internal_variables = new Variables(m_mesh,name(),itemKind(),var_count_name,
                                         var_data_name,var_unique_ids_name,var_groups_name,
                                         var_current_id_name,var_new_owner_name,
                                         var_parent_mesh_name,var_parent_family_name,
                                         var_parent_family_depth_name,
                                         var_child_meshes_name,
                                         var_child_families_name);
    m_items_data = &m_internal_variables->m_items_data._internalTrueData()->_internalDeprecatedValue();
    m_items_data->reserve(1000);
    m_items_unique_id = &m_internal_variables->m_items_unique_id._internalTrueData()->_internalDeprecatedValue();
    m_items_unique_id_view = m_items_unique_id->view();
    //m_items_unique_ids->reserve(1000);
    //m_variables->m_current_id = 0;
  }

  // Pour pouvoir remettre à jour les ItemSharedInfos après relecture
  m_observers.addObserver(this,
                          &ItemFamily::_notifyDataIndexChanged,
                          m_internal_variables->m_items_data.variable()->readObservable());

  m_variable_synchronizer = ParallelMngUtils::createSynchronizerRef(pm,this);

  m_item_sort_function = _defaultItemSortFunction();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
nbItem() const
{
  return m_infos.nbItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemFamily::
maxLocalId() const
{
  return m_infos.maxUsedLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalList ItemFamily::
itemsInternal()
{
  return m_infos.itemsInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily * ItemFamily::
parentFamily() const
{
  return m_parent_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
setParentFamily(IItemFamily * parent)
{
  m_parent_family = parent;
  if (parent == this) // Auto-référencement
    m_parent_family_depth = 1;
  else if (!parent) // Pas de parent
    m_parent_family_depth = 0;
  else { // Référencement croisé
    m_parent_family_depth = parent->parentFamilyDepth()+1;
    m_parent_family->addChildFamily(this);
  }
  ARCANE_ASSERT((m_parent_family_depth < 2),("Not test if more than one depth level"));
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
  for(Integer i=0;i<m_child_families.size(); ++i) {
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
 * Méthode appelée par le maillage à la fin d'un IMesh::endUpdate().
 * Cette méthode est collective et permet donc de faire des opérations
 * collectives une fois les modifications de maillage terminées.
 */
void ItemFamily::
notifyEndUpdateFromMesh()
{
  // Recalcul les infos de connectivités globales à tous les sous-domaines
  m_global_connectivity_info->fill(m_item_shared_infos);
  m_global_connectivity_info->reduce(parallelMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
endUpdate()
{
  _endUpdate(true);
  // TODO: tester la connectivité mais ca fait planter certains cas tests
  // (dof, amr2 et mesh_modification). A voir si c'est normal.
  //if (arcaneIsCheck())
  //checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
endAllocate()
{
  // La variable n'est pas "used" par défaut car la famille n'est pas encore prête.
  // Sur les sous-familles, il suffit donc de filtrer setUsed au moment du endAllocate
  if (!m_parent_family) {
    m_internal_variables->setUsed();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemFamily::
_partialEndUpdate()
{
  bool need_end_update = m_infos.changed();
  info(4) << "ItemFamily::endUpdate() " << fullName() << " need_end_update?=" << need_end_update;
  if (!need_end_update){
    // Même si aucune entité n'est ajoutée ou supprimée, si \a m_need_prepare_dump
    // est vrai cela signifie que quelque chose a changé dans la famille. Dans ce
    // cas il est préférable d'incrémenter le current_id. Sans cela, si on appelle
    // readFromDump() (par exemple suite à un retour-arrière), les données ne seront
    // pas restaurées si entre temps current_id n'a pas changé.
    if (m_need_prepare_dump){
      m_local_connectivity_info->fill(m_item_shared_infos);
      ++m_current_id;
    }
    return true;
  }
  m_item_need_prepare_dump = true;
  m_local_connectivity_info->fill(m_item_shared_infos);
  ++m_current_id;
  m_internal_variables->m_items_data.variable()->syncReferences();
  m_items_data = &m_internal_variables->m_items_data._internalTrueData()->_internalDeprecatedValue();

  // Update "external" connectivities
  if (m_connectivity_mng)
    m_connectivity_mng->setModifiedItems(this,m_infos.addedItems(), m_infos.removedItems());
  //
  m_infos.finalizeMeshChanged();

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_endUpdate(bool need_check_remove)
{
  if (_partialEndUpdate())
    return;

  resizeVariables(false);
  info(4) << "ItemFamily:endUpdate(): " << fullName()
          << " hashmapsize=" << itemsMap().buckets().size()
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
  _updateGroup(group,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateGroup(ItemGroup group,bool need_check_remove)
{
  // Pas besoin de recalculer le groupe des entités globales
  if (group==m_infos.allItems())
    return;

  if (group.internal()->hasComputeFunctor())
    group.invalidate();
  if (need_check_remove){
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
  for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
    ItemGroup group = *i;
    _updateGroup(group,need_check_remove);
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
resizeVariables(bool force_resize)
{
  debug(Trace::High) << "ItemFamily::resizeVariables: name=" << fullName()
                     << " varsize=" << maxLocalId()
                     << " nb_item=" << nbItem()
                     << " currentsize=" << m_current_variable_item_size;
  if (force_resize || (maxLocalId()!=m_current_variable_item_size)){
    info(4) << "ItemFamily::resizeVariables: name=" << fullName()
            << " varsize=" << maxLocalId()
            << " nb_item=" << nbItem()
            << " currentsize=" << m_current_variable_item_size
            << " group_nb_item=" << allItems().size()
            << " nb_var=" << m_used_variables.size();

    m_current_variable_item_size = maxLocalId();

    for( IVariable* var : m_used_variables ){
      _updateVariable(var);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(ArrayView<Int64> ids,bool do_fatal) const
{
  m_infos.itemsUniqueIdToLocalId(ids,do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       Int64ConstArrayView unique_ids,
                       bool do_fatal) const
{
  m_infos.itemsUniqueIdToLocalId(local_ids,unique_ids,do_fatal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       ConstArrayView<ItemUniqueId> unique_ids,
                       bool do_fatal) const
{
  m_infos.itemsUniqueIdToLocalId(local_ids,unique_ids,do_fatal);
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
  return m_infos.allItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
createGroup(const String& name,Int32ConstArrayView elements,bool do_override)
{
  debug(Trace::High) << "ItemFamily:createGroup(name,Int32ConstArrayView): " << m_name << ":"
                     << " group_name=" << name
                     << " count=" << elements.size()
                     << " override=" << do_override;

  _checkNeedEndUpdate();

  {
    ItemGroup group;
    if (do_override){
      group = findGroup(name);
      if (group.null())
        group = createGroup(name);
    }
    else{
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
  // SDP: probleme protection-reprise ...
  {
    ItemGroup g = findGroup(name);
    if (!g.null()) {
      fatal() << "Attempting to create an already existing group '" << name << "'";
    }
  }
  debug() << "ItemFamily:createGroup(name): " << m_name << ":"
          << " name=" << name;
  ItemGroup group(new ItemGroupImpl(this,name));
  _processNewGroup(group);
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
createGroup(const String& name,const ItemGroup& parent,bool do_override)
{
  ItemGroup group = findGroup(name);
  if (!group.null()) {
    if (do_override){
      if (group.internal()->parentGroup()!=parent)
        fatal() << "Group already existing but with a different parent";
      if (group==parent)
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
  group = ItemGroup(new ItemGroupImpl(this,parent.internal(),name));
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
  for( ItemGroupList::Enumerator i(current_groups); ++i; ){
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
  for( ItemGroup group : m_item_groups ){
    if (m_is_parallel && group.internal()->hasComputeFunctor())
      group.invalidate();
  }

  // Propage les modifications sur les sous-familles
  for(Integer i=0;i<m_child_families.size(); ++i){
    IItemFamily * family = m_child_families[i];
    ItemInternalArrayView items(family->itemsInternal());
    for( Integer z=0, zs=items.size(); z<zs; ++z ){
      ItemInternal* item = items[z];
      if (item->isSuppressed())
        continue;
      Item parent_item = item->parent(0);
      ARCANE_ASSERT((parent_item.uniqueId() == item->uniqueId()),("Inconsistent parent uid"));
      item->setOwner(parent_item.owner(), m_sub_domain_id);
    }
    family->notifyItemsOwnerChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_processNewGroup(ItemGroup group)
{
  // Vérifie que le groupe à le bon type
  if (group.itemKind()!=itemKind()){
    ARCANE_FATAL("Incoherent family name={0} wanted={1} current={2}",
                 fullName(),itemKind(),group.itemKind());
  }
  m_item_groups.add(group);
  m_need_prepare_dump = true;
  // En séquentiel, tous les groupes sont des groupes propres
  // TODO: regarder pour supprimer le test avec 'm_is_parallel' mais
  // si on le fait ca fait planter certains tests avec aleph_kappa.
  if (!m_is_parallel && m_mesh->meshPartInfo().nbPart()==1)
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
  for( const ItemGroup& group : m_item_groups ){
    if (group.name()==name)
      return group;
  }
  return ItemGroup();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemFamily::
findGroup(const String& name,bool create_if_needed)
{
  ItemGroup group = findGroup(name);
  if (group.null()){
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
  if (m_infos.changed())
    ARCANE_FATAL("missing call to endUpdate() after modification");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_setSharedInfosBasePtr()
{
  Int32* new_ptr = m_items_data->data();
  m_item_shared_infos->setSharedInfosPtr(new_ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_setDataIndexForItem(ItemInternal* item,Int32 data_index)
{
  item->setDataIndex(data_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_setSharedInfoForItem(ItemInternal* item,ItemSharedInfo* isi,Int32 data_index)
{
  item->setSharedInfo(isi);
  _setDataIndexForItem(item,data_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Modifie le ItemInternal::sharedInfo() en considérant que seul
 * le nombre d'entités connecté change.
 *
 * Dans ce cas, il faut uniquement mettre à jour le nombre d'entités
 * connectées si on utilise les anciennes connectivités.
 */
void ItemFamily::
_setSharedInfosNoCopy(ItemInternal* item,ItemSharedInfo* isi)
{
  item->setSharedInfo(isi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
prepareForDump()
{
  info(4) << "ItemFamily::prepareFromDump(): " << fullName()
          << " need=" << m_need_prepare_dump
          << " item-need=" << m_item_need_prepare_dump
          << " m_item_shared_infos->hasChanged()=" << m_item_shared_infos->hasChanged()
          << " nb_item=" << m_infos.nbItem();

  // TODO: ajoute flag vérification si nécessaire
  if (m_item_need_prepare_dump || m_item_shared_infos->hasChanged()){
    info(4) << "Prepare for dump:2: name=" << m_name << " nb_alloc=" << m_nb_allocate_info
            << " uid_size=" << m_items_unique_id->size() << " cap=" << m_items_unique_id->capacity()
            << " byte=" << m_items_unique_id->capacity()*sizeof(Int64);

    //TODO: pouvoir spécifier si on souhaite compacter ou pas.
    compactItems(false);

    compactReferences();

    // Suppose compression
    m_infos.prepareForDump();
    m_item_shared_infos->prepareForDump();
    m_need_prepare_dump = true;
  }
  m_item_need_prepare_dump = false;
  if (m_need_prepare_dump){
    m_internal_variables->m_current_id = m_current_id;
    debug() << " SET FAMILY ID name=" << name() << " id= " << m_current_id
            << " saveid=" << m_internal_variables->m_current_id();
    ItemInternalList items(m_infos.itemsInternal());
    Integer nb_item = m_infos.nbItem();
    m_internal_variables->m_items_shared_data_index.resize(nb_item);
    IntegerArrayView items_shared_data_index(m_internal_variables->m_items_shared_data_index);
    debug() << "ItemFamily::prepareForDump(): " << m_name
            << " count=" << nb_item << " currentid=" << m_current_id;
    for( Integer i=0; i<nb_item; ++i ){
      ItemInternal* item = items[i];
      items_shared_data_index[i] = item->sharedInfo()->index();
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

    // Données de liaison de familles
    {
      if (m_parent_family) {
        m_internal_variables->m_parent_family_name = m_parent_family->name();
        m_internal_variables->m_parent_mesh_name = m_parent_family->mesh()->name();
      }
      m_internal_variables->m_parent_family_depth = m_parent_family_depth;
      const Integer child_count = m_child_families.size();
      m_internal_variables->m_child_meshes_name.resize(child_count);
      m_internal_variables->m_child_families_name.resize(child_count);
      for(Integer i=0;i<child_count;++i) {
        m_internal_variables->m_child_meshes_name[i] = m_child_families[i]->mesh()->name();
        m_internal_variables->m_child_families_name[i] = m_child_families[i]->name();
      }
    }

    {
      // Détermine le nombre de groupes et d'entités à sauver.
      // On ne sauve pas les groupes générés dynamiquement
      Integer nb_group_to_save = 0;
      for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
        const ItemGroup& group = *i;
        if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
          continue;
        debug(Trace::High) << "Save group info name=" << group.name();
        ++nb_group_to_save;
      }
      m_internal_variables->m_groups_name.resize(nb_group_to_save);
      {
        Integer current_group_index = 0;
        for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
          const ItemGroup& group = *i;
          if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
            continue;
          m_internal_variables->m_groups_name[current_group_index] = group.name();
          ++current_group_index;
        }
      }
    }
  }
  // Fait en sorte que les groupes soient à jour, pour être sur que
  // la sauvegarde sera correcte.
  // NOTE: Est-ce ici qu'il faut le faire ?
  // NOTE: plutot utiliser un observer sur la variable du groupe?
  for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
    ItemGroup group = *i;
    // Pas besoin de recalculer le groupe des entités globales
    if (group==m_infos.allItems())
      continue;
    group.internal()->checkNeedUpdate();
  }

  m_need_prepare_dump = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
readFromDump()
{
  //TODO: GG: utiliser un flag pour indiquer qu'il faudra reconstruire
  // les infos de synchro mais ne pas le faire directement dans cette methode.

  // Le numéro de la partie peut changer en reprise. Il faut donc le
  // remettre à jour. De même, si on passe d'un maillage séquentiel à un
  // maillage en plusieurs parties en reprise, il faut supprimer le isOwn()
  // du groupe de toutes les entités.
  const MeshPartInfo& part_info = m_mesh->meshPartInfo();
  m_sub_domain_id = part_info.partRank();
  if (m_infos.allItems().isOwn() && part_info.nbPart()>1)
    m_infos.allItems().setOwn(false);

  m_items_unique_id_view = m_items_unique_id->view();
  // NOTE: l'implémentation actuelle suppose que les dataIndex() des
  // entites sont consécutifs et croissants avec le localId() des entités
  // (c.a.d l'entité de localId() valant 0 à aussi un dataIndex() de 0,
  // celle de localId() valant 1, le dataIndex() suivant ...)
  // Cette condition est vrai si compactReferences() a été appelé.
  // Lorsque ce ne sera plus le cas (trou dans la numérotation), il faudra
  // ajouter une variable data_index sur les entités.
  IntegerArrayView items_shared_data_index(m_internal_variables->m_items_shared_data_index);
  Integer nb_item = items_shared_data_index.size();
  info(4) << "ItemFamily::readFromDump(): " << fullName()
          << " count=" << nb_item
          << " data=" << m_internal_variables->m_items_data.size()
          << " currentid=" << m_current_id
          << " saveid=" << m_internal_variables->m_current_id();

  if (m_internal_variables->m_current_id()==m_current_id){
    debug() << "Family unchanged. Nothing to do.";
    //GG: il faut quand meme recalculer les infos de synchro car cette famille
    // sur les autres sous-domaine peut avoir changee et dans ce cas cette
    // fonctione sera appelee. De meme, la liste des groupes peut avoir changée
    // et leur valeur aussi pour les groupes recalculés donc il faut les invalider
    _checkComputeSynchronizeInfos(0);
    _readGroups();
    _invalidateComputedGroups();
    return;
  }

  m_current_id = m_internal_variables->m_current_id();
  // IMPORTANT: remise à zéro pour obligatoirement redimensionner les variables
  // au prochain ajout d'entités.
  m_current_variable_item_size = 0;

  // Données de liaison de famille
  {
    IMeshMng* mesh_mng = m_mesh->meshMng();
    if (!m_internal_variables->m_parent_mesh_name().null()) {
      IMesh* parent_mesh = mesh_mng->findMeshHandle(m_internal_variables->m_parent_mesh_name()).mesh();
      m_parent_family = parent_mesh->findItemFamily(m_internal_variables->m_parent_family_name(),true); // true=> fatal si non trouvé
    }
    m_parent_family_depth = m_internal_variables->m_parent_family_depth();
    ARCANE_ASSERT((m_internal_variables->m_child_meshes_name.size() == m_internal_variables->m_child_families_name.size()),
                  ("Incompatible child mesh/family sizes"));
    Integer child_count = m_internal_variables->m_child_families_name.size();
    for(Integer i=0;i<child_count;++i) {
      IMesh* child_mesh = mesh_mng->findMeshHandle(m_internal_variables->m_child_meshes_name[i]).mesh();
      IItemFamily * child_family = child_mesh->findItemFamily(m_internal_variables->m_child_families_name[i],true); // true=> fatal si non trouvé
      m_child_families.add(dynamic_cast<ItemFamily*>(child_family));
    }
  }

  m_item_shared_infos->readFromDump();
  _setSharedInfosBasePtr();
  m_infos.readFromDump();

  // En relecture les entités sont compactées donc la valeur max du localId()
  // est égal au nombre d'entité.
  {
    ArrayView<ItemSharedInfo*> item_shared_infos = m_item_shared_infos->itemSharedInfos();
    ItemInternalList items(m_infos.itemsInternal());
    Integer data_index = 0;
    for( Integer i=0; i<nb_item; ++i ){
      Integer shared_data_index = items_shared_data_index[i];
      ItemSharedInfo* isi = item_shared_infos[shared_data_index];
      Int64 uid = (*m_items_unique_id)[i];
      ItemInternal* item = m_infos.allocOne(uid);
      _setSharedInfoForItem(item,isi,data_index);
      data_index += isi->neededMemory();
    }
  }
  // Supprime les entités du groupe total car elles vont être remises à jour
  // lors de l'appel à _endUpdate()
  m_infos.allItems().clear();

  // Notifie les connectivités sources qu'on vient de faire une relecture.
  for( IIncrementalItemConnectivity* c : m_source_incremental_item_connectivities )
    c->notifyReadFromDump();

  // Recréation des groupes si nécessaire
  _readGroups();

  // Invalide les groupes recalculés
  _invalidateComputedGroups();

  _endUpdate(false);

  _checkComputeSynchronizeInfos(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_invalidateComputedGroups()
{
  // Si le groupe a un parent, il n'a pas de variable associée et
  // de plus peut contenir des valeurs invalides suite à un retour-arrière.
  // Dans ce cas, on le vide et on l'invalide.
  for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
    ItemGroup group = *i;
    if (!group.internal()->parentGroup().null()){
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
  for( Integer i=0, is=groups_var.size(); i<is; ++i ){
    String name(groups_var[i]);
    debug() << "Readign group again: " << name;
    ItemGroup group = findGroup(name);
    if (group.null())
      createGroup(name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test collectif permettant de savoir s'il faut mettre
 * à jour les infos de synchro.
 *
 * \a changed vaut 0 si pas de mise à jour, 1 sinon.
 */
void ItemFamily::
_checkComputeSynchronizeInfos(Int32 changed)
{
  Int32 global_changed = parallelMng()->reduce(Parallel::ReduceMax,changed);
  if (global_changed!=0)
    computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compacte les entités.
 */
void ItemFamily::
compactItems(bool do_sort)
{
  _compactItems(do_sort);

  // Il est necessaire de mettre a jour les groupes.
  // TODO verifier s'il faut le faire tout le temps
  m_need_prepare_dump = true;

  // Indique aussi qu'il faudra refaire un compactReference()
  // lors du dump.
  // NOTE: spécifier cela forcera aussi un recompactage lors du prepareForDump()
  // et ce compactage est inutile dans le cas présent.
  // TODO: regarder comment indiquer au prepareForDump() qu'on souhaite
  // juste faire un compactReference().
  m_item_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compacte les entités.
 */
void ItemFamily::
_compactItems(bool do_sort)
{
  IMeshCompactMng* compact_mng = mesh()->_compactMng();
  IMeshCompacter* compacter = compact_mng->beginCompact(this);

  try{
    compacter->setSorted(do_sort);
    compacter->doAllActions();
  }
  catch(...){
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
  m_infos.beginCompactItems(compact_infos);

  if (arcaneIsCheck())
    m_infos.checkValid();

  Int32ConstArrayView new_to_old_ids = compact_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = compact_infos.oldToNewLocalIds();

  for( IIncrementalItemConnectivity* c : m_source_incremental_item_connectivities )
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for( IIncrementalItemConnectivity* c : m_target_incremental_item_connectivities )
    c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  for (IItemConnectivity* c : m_source_item_connectivities )
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for( IItemConnectivity* c : m_target_item_connectivities )
      c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  if (m_connectivity_mng)
    m_connectivity_mng->notifyLocalIdChanged(this,old_to_new_ids, nbItem());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
finishCompactItems(ItemFamilyCompactInfos& compact_infos)
{
  if (arcaneIsCheck())
    m_infos.checkValid();

  m_infos.finishCompactItems(compact_infos);

  for( ItemConnectivitySelector* ics : m_connectivity_selector_list )
    ics->compactConnectivities();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Copie les valeurs des entités numéros @a source dans les entités
 * numéro @a destination
 *
 * @param source liste des @b localId source
 * @param destination liste des @b localId destination
 */
void ItemFamily::
copyItemsValues(Int32ConstArrayView source,Int32ConstArrayView destination)
{
  ARCANE_ASSERT(source.size()==destination.size(),
		("Can't copy. Source and destination have different size !"));

  if (source.size()!=0){
    for( IVariable* var : m_used_variables ){
      // (HP) : comme vu avec Gilles et Stéphane, on ne met pas de filtre à ce niveau
      // // si la variable est temporaire ou no restore, on ne la copie pas
      // if (!(var->property() & (IVariable::PTemporary | IVariable::PNoRestore))) {
      //if (var->itemFamily()==this) {
      var->copyItemsValues(source,destination);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Copie les moyennes des valeurs des entités numéros
 * @a first_source et @a second_source dans les entités numéros
 * @a destination
 *
 * @param first_source liste des @b localId de la 1ère source
 * @param second_source  liste des @b localId de la 2ème source
 * @param destination  liste des @b localId destination
 */
void ItemFamily::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  ARCANE_ASSERT(first_source.size()==destination.size(),
		("Can't copy. : first_source and destination have different size !"));
  ARCANE_ASSERT(second_source.size()==destination.size(),
		("Can't copy : second_source and destination have different size !"));

  if (first_source.size() != 0) {
    for( IVariable* var : m_used_variables ){
      if (!(var->property() & (IVariable::PTemporary | IVariable::PNoRestore))){
        var->copyItemsMeanValues(first_source,second_source,destination);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compacte les variables et les groupes.
 *
 * \warning: Cette méthode doit être appelée durant un compactage
 * (entre un appel à m_infos.beginCompactItems() et m_infos.endCompactItems()).
 */
void ItemFamily::
compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos)
{
  Int32ConstArrayView new_to_old_ids = compact_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = compact_infos.oldToNewLocalIds();

  for( IVariable* var : m_used_variables ){
    debug(Trace::High) << "Compact variable " << var->fullName();
    var->compact(new_to_old_ids);
  }

  m_variable_synchronizer->changeLocalIds(old_to_new_ids);

  for( ItemGroupList::Enumerator i(m_item_groups); ++i; ){
    ItemGroup group = *i;
    debug(Trace::High) << "Change group Ids: " << group.name();
    group.internal()->changeIds(old_to_new_ids);
    if (group.hasSynchronizer())
      group.synchronizer()->changeLocalIds(old_to_new_ids);
  }

  for(Integer i=0;i<m_child_families.size();++i)
    m_child_families[i]->_compactFromParentFamily(compact_infos);

  info(4) << "End compact family=" << fullName()
          << " max_local_id=" << maxLocalId()
          << " nb_item=" << nbItem();

  // Apres compactage, les variables seront allouées avec comme nombre
  // d'éléments le nombre d'entité (c'est dans DynamicMeshKindInfos::finishCompactItems()
  // que maxLocalId() devient égal à nbItem()).
  m_current_variable_item_size = nbItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compacte les connectivités.
 *
 * \warning: Cette méthode doit être appelée durant un compactage
 * (entre un appel à m_infos.beginCompactItems() et m_infos.endCompactItems()).
 */
#ifdef NEED_MERGE
void ItemFamily::
compactConnectivities()
{
  Int32ConstArrayView new_to_old_ids = m_infos.newToOldLocalIds();
  Int32ConstArrayView old_to_new_ids = m_infos.oldToNewLocalIds();
  for (IItemConnectivity* c : m_source_connectivities)
    c->notifySourceFamilyLocalIdChanged(new_to_old_ids);

  for( IItemConnectivity* c : m_target_connectivities )
      c->notifyTargetFamilyLocalIdChanged(old_to_new_ids);

  if (m_connectivity_mng)
    m_connectivity_mng->notifyLocalIdChanged(this,old_to_new_ids, nbItem());
}
#endif


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_compactFromParentFamily(const ItemFamilyCompactInfos& compact_infos)
{
  debug() << "Compacting child family " << fullName();
  if (m_parent_family_depth>1)
    throw NotImplementedException(A_FUNCINFO,"Too deep parent family: not yet implemented");
  Int32ConstArrayView old_to_new_lids(compact_infos.oldToNewLocalIds());
  ARCANE_ASSERT((nbItem()==0 || !old_to_new_lids.empty()),("Empty oldToNewLocalIds"));
  debug() << "\tfrom parent family " << m_parent_family->name();
  if (this == m_parent_family)
    return; // already self compacted
  ItemInternalArrayView items(itemsInternal());
  for( Integer z=0, zs=items.size(); z<zs; ++z ){
    ItemInternal* item = items[z];
    Int32 * parentPtr = item->parentPtr();
    Int32 old_parent_lid = parentPtr[0]; // depth==1 only !!
    parentPtr[0] = old_to_new_lids[old_parent_lid];
  }
  // Si depth>1, il faudrait plutot propager le compactage en modifiant les
  // oldToNewLocalIds des familles du sous-maillage courant et en appelant
  // DynamicMesh::_compactItems en cascade (à partir de ce sous-maillage)
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addItems(Int64ConstArrayView unique_ids,Int32ArrayView items)
{
  ARCANE_UNUSED(unique_ids);
  ARCANE_UNUSED(items);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addItems(Int64ConstArrayView unique_ids,ArrayView<Item> items)
{
  ARCANE_UNUSED(unique_ids);
  ARCANE_UNUSED(items);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addItems(Int64ConstArrayView unique_ids,ItemGroup items)
{
  ARCANE_UNUSED(unique_ids);
  ARCANE_UNUSED(items);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost)
{
  ARCANE_UNUSED(local_ids);
  ARCANE_UNUSED(keep_ghost);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
exchangeItems()
{
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
mergeItems(Int32 local_id1, Int32 local_id2)
{
  ARCANE_UNUSED(local_id1);
  ARCANE_UNUSED(local_id2);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemFamily::
getMergedItemLID(Int32 local_id1, Int32 local_id2)
{
  ARCANE_UNUSED(local_id1);
  ARCANE_UNUSED(local_id2);
  throw NotSupportedException(A_FUNCINFO,
                              "this kind of family doesn't support this operation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
compactReferences()
{
  Integer old_mem = m_items_data->size();

  //TODO: il faut prendre la mémoire réellement utilisée et
  // pas la mémoire necessaire
  ItemInternalList items(m_infos.itemsInternal());
  Integer nb_item = items.size();
  Integer needed_memory = 0;
  for( Integer i=0; i<nb_item; ++i ){
    needed_memory += items[i]->neededMemory();
  }

  info(4) << "CompactRefererences: family=" << fullName()
          << " old=" << old_mem << " new=" << needed_memory;

  Int32UniqueArray new_data;
  new_data.resize(needed_memory);
  Int32* new_data_ptr = new_data.data();
  Integer current_index = 0;
  for( Integer i=0; i<nb_item; ++i ){
    ItemInternal* item = items[i];
    Integer nb = item->neededMemory();
    item->_internalCopyAndSetDataIndex(new_data_ptr,current_index);
    current_index += nb;
  }
  m_items_data->copy(new_data);

  _setSharedInfosBasePtr();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_checkValid()
{
  // Vérifie que le numéro de la partie est le même que celui du maillage
  {
    Int32 part_rank = m_mesh->meshPartInfo().partRank();
    if (m_sub_domain_id!=part_rank)
      ARCANE_FATAL("Family {0} Bad value for partRank ({1}) expected={2}",
                   fullName(),m_sub_domain_id,part_rank);
  }
  // Vérifie que 'm_items_data' et 'm_internal_variables->m_items_data'
  // sont les mêmes
  {
    Int32* i1 = m_items_data->data();
    Int32* i2 = m_internal_variables->m_items_data._internalTrueData()->_internalDeprecatedValue().data();
    if (i1!=i2){
      fatal() << "ItemFamily: " << m_name
              << ": items_data invalid ptr1=" << i1 << " ptr2=" << i2;
    }
  }

  // Vérifie que la famille est valide.
  // 1. Vérifie que chaque entité à un sharedInfo() dont le pointeur
  //    contenant le tableau des info est identique à \a m_items_data->begin()
  ItemInternalList items(m_infos.itemsInternal());
  Int32* infos_begin = m_items_data->data();
  Integer nb_error = 0;
  for( Integer i=0, is=items.size(); i<is; ++i ){
    ItemInternal* item = items[i];
    ItemSharedInfo* isi = item->sharedInfo();
    if (isi->_infos()!=infos_begin){
      if (nb_error<10)
        error() << "ItemFamily: Info shared by an invalid entity"
                << " LID=" << i << " Item=" << item
                << " Shared=" << isi << " Begin=" << isi->_infos()
                << " (expected:" << infos_begin;
      ++nb_error;
    }
    //TODO: verifier que 'isi' est bien dans la table de hashage
  }
  if (nb_error!=0){
    m_item_shared_infos->dumpSharedInfos();
    fatal() << "ItemFamily: " << m_name << ": " << nb_error
            << " errors: invalid internal structure";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_reserveInfosMemory(Integer memory)
{
  Int32* old_ptr = m_items_data->data();
  m_items_data->reserve(memory);
  Int32* new_ptr = m_items_data->data();
  if (old_ptr!=new_ptr){
    info(4) << "RESIZE_ONE1 Size=" << m_items_data->size() << " capacity=" << m_items_data->capacity()
            << " ptr=" << new_ptr << " name=" << m_name
            << " var_size=" << m_internal_variables->m_items_data.size()
            << " var_capacity=" << m_internal_variables->m_items_data._internalTrueData()->capacity()
            << " ptr=" << m_internal_variables->m_items_data.data();
    _setSharedInfosBasePtr();
    _checkValid();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_resizeInfos(Integer new_size)
{
  Int32* old_ptr = m_items_data->data();
  Integer old_size = m_items_data->size();
  Integer old_capacity = m_items_data->capacity();
  if (new_size>old_capacity){
    if (new_size>5000000)
      m_items_data->reserve((Integer)(new_size * 1.2));
    else if (new_size>500000)
      m_items_data->reserve((Integer)(new_size * 1.5));
    else
      m_items_data->reserve((Integer)(new_size * 2.0));
  }
  m_items_data->resize(new_size);
  Int32* new_ptr = m_items_data->data();
  if (old_ptr!=new_ptr){
    info(4) << "RESIZE_ONE2 OldSize=" << old_size << " new_size=" << new_size
            << " capacity=" << m_items_data->capacity()
            << " ptr=" << new_ptr << " name=" << m_name
            << " var_size=" << m_internal_variables->m_items_data.size()
            << " var_capacity=" << m_internal_variables->m_items_data._internalTrueData()->capacity()
            << " ptr=" << m_internal_variables->m_items_data.data()
            << " old_size=" << old_size
            << " nb_item=" << m_infos.nbItem()
            << " max_local_id=" << maxLocalId();
    _setSharedInfosBasePtr();
    _checkValid();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
_allocMany(Integer memory)
{
  Integer s = m_items_data->size();
  _resizeInfos(s+memory);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo* ItemFamily::
_findSharedInfo(ItemTypeInfo* type)
{
  ItemSharedInfo* isi = m_item_shared_infos->findSharedInfo(type);
  isi->_setInfos(m_items_data->data());
  return isi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_copyInfos(ItemInternal* item,ItemSharedInfo* old_isi,ItemSharedInfo* new_isi)
{
  // Signale qu'il faudra compacter les entités au moment du dump
  m_item_need_prepare_dump = true;

  Integer new_data_index = _allocMany(new_isi->neededMemory());
  item->_internalCopyAndChangeSharedInfos(old_isi,new_isi,new_data_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateSharedInfoAdded(ItemInternal* item)
{
  ItemSharedInfo* old_isi = item->sharedInfo();
  ItemSharedInfo* new_isi = _findSharedInfo(old_isi->m_item_type);
  _setSharedInfosNoCopy(item,new_isi);

  m_need_prepare_dump = true;
  new_isi->addReference();
  old_isi->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateSharedInfoRemoved4(ItemInternal* item)
{
  // TODO: regarder pourquoi _updateSharedInfoRemoved7() n'a pas le même code.
  m_need_prepare_dump = true;

  ItemSharedInfo* old_isi = item->sharedInfo();
  ItemSharedInfo* new_isi = _findSharedInfo(old_isi->m_item_type);
  _setSharedInfosNoCopy(item,new_isi);

  new_isi->addReference();
  old_isi->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_updateSharedInfoRemoved7(ItemInternal*)
{
  // TODO: regarder fusion possible avec les autres surcharges de _updateSharedInfo
  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> inline bool
_checkResizeArray(Array<DataType>& array,Integer new_max_index)
{
  Integer s = array.size();
  if (new_max_index>=s){
    Integer new_size = new_max_index + 1;
    if (new_size>array.capacity()){
      if (new_size>5000000)
        array.reserve((Integer)(new_size * 1.2));
      else if (new_size>500000)
        array.reserve((Integer)(new_size * 1.5));
      else
        array.reserve((Integer)(new_size * 2.0));
    }
    array.resize(new_size);
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_setUniqueId(Int32 lid,Int64 uid)
{
  _checkResizeArray(*m_items_unique_id,lid);
  (*m_items_unique_id)[lid] = uid;
  m_items_unique_id_view = m_items_unique_id->view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_allocateInfos(ItemInternal* item,Int64 uid,ItemTypeInfo* type)
{
  ItemSharedInfo* isi = _findSharedInfo(type);
  _allocateInfos(item,uid,isi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_allocateInfos(ItemInternal* item,Int64 uid,ItemSharedInfo* isi)
{
  // TODO: faire en même temps que le réalloc de la variable uniqueId()
  //  le réalloc des m_source_incremental_item_connectivities.
  Int32 local_id = item->localId();
  _setUniqueId(local_id,uid);
  // Il faut positionner le ItemSharedInfo avant le _allocMany
  // sinon les tests de vérification échouent.
  item->setSharedInfo(isi);
  Integer new_data_index = _allocMany(isi->neededMemory());
  _setSharedInfoForItem(item,isi,new_data_index);
  
  item->reinitialize(uid,m_default_sub_domain_owner,m_sub_domain_id);
  ++m_nb_allocate_info;
  // Notifie les connectivitées incrémentales qu'on a ajouté un élément à la source
  for( IIncrementalItemConnectivity* c : m_source_incremental_item_connectivities )
    c->notifySourceItemAdded(ItemLocalId(local_id));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_notifyDataIndexChanged()
{
  //warning() << "Data Index changed ! " << m_name;
  m_items_unique_id_view = m_items_unique_id->view();
  _setSharedInfosBasePtr();
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
  // Ne calcul les infos de synchronisation que si on est en parallèle
  // et que le nombre de parties est égal au nombre de rang du parallelMng(),
  // ce qui n'est pas le cas en reprise lors d'un changement du nombre de
  // sous-domaines.
  if (m_is_parallel && m_mesh->meshPartInfo().nbPart()==parallelMng()->commSize()){
    m_variable_synchronizer->compute();
    _updateItemsSharedFlag();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO déplacer cette méthode en dehors de ItemFamily
void ItemFamily::
reduceFromGhostItems(IVariable* v,IDataOperation* operation)
{
  if (!v)
    return;
  if (v->itemFamily()!=this && v->itemGroup().itemFamily()!=this)
    throw ArgumentException(A_FUNCINFO,"Variable not in this family");
  Parallel::GhostItemsVariableParallelOperation op(this);
  op.setItemFamily(this);
  op.addVariable(v);
  op.applyOperation(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO déplacer cette méthode en dehors de ItemFamily
void ItemFamily::
reduceFromGhostItems(IVariable* v,Parallel::eReduceType reduction)
{
  ScopedPtrT<IDataOperation> operation;
  operation = v->dataFactoryMng()->createDataOperation(reduction);
  reduceFromGhostItems(v,operation.get());
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
findVariable(const String& var_name,bool throw_exception)
{
  IVariableMng* vm = subDomain()->variableMng();
  StringBuilder vname = mesh()->name();
  vname += "_";
  vname += name();
  vname += "_";
  vname += var_name;
  IVariable* var = vm->findVariableFullyQualified(vname.toString());
  if (!var && throw_exception){
    ARCANE_FATAL("No variable named '{0}' in family '{1}'",var_name,name());
  }
  return var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
clearItems()
{
  m_infos.clear();

  endUpdate();

  // Compacte les références pour économiser la mémoire.
  compactReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
checkUniqueIds(Int64ConstArrayView unique_ids)
{
  Integer nb_item = unique_ids.size();
  Int64UniqueArray all_unique_ids;
  IParallelMng* pm = m_mesh->parallelMng();
  pm->allGatherVariable(unique_ids,all_unique_ids);
  HashTableMapT<Int64,Integer> items_map(nb_item*2,true);
  info() << "ItemFamily::checkUniqueIds name=" << name() << " n=" << nb_item
         << " total=" << all_unique_ids.size();
  for( Integer i=0; i<nb_item; ++i )
    items_map.add(unique_ids[i],0);
  for( Integer i=0, is= all_unique_ids.size(); i<is; ++i ){
    HashTableMapT<Int64,Integer>::Data* data = items_map.lookup(all_unique_ids[i]);
    if (data)
      ++data->value();
  }
  for( Integer i=0; i<nb_item; ++i ){
    Integer nb_ref = items_map[unique_ids[i]];
    if (nb_ref!=1){
      fatal() << "Duplicate unique_id=" << unique_ids[i];
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup ItemFamily::
findAdjencyItems(const ItemGroup& group,const ItemGroup& sub_group,
                 eItemKind link_kind,Integer layer)
{
  AdjencyInfo at(group,sub_group,link_kind,layer);
  AdjencyGroupMap::const_iterator i = m_adjency_groups.find(at);

  if (i==m_adjency_groups.end()){
    debug() << "** BUILD ADJENCY_ITEMS : " << group.name() << " x "
            << sub_group.name() << " link=" << link_kind << " nblayer=" << layer;
    ItemPairGroup v(new ItemPairGroupImpl(group,sub_group));
    mesh()->utilities()->computeAdjency(v,link_kind,layer);
    m_adjency_groups.insert(std::make_pair(at,v));
    return v;
  }
  debug() << "** FOUND KNOWN ADJENCY_ITEMS! : " << group.name() << " x "
          << sub_group.name() << " link=" << link_kind << " nblayer=" << layer;
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxNodePerItem() const
{
  return m_local_connectivity_info->maxNodePerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxEdgePerItem() const
{
  return m_local_connectivity_info->maxEdgePerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxFacePerItem() const
{
  return m_local_connectivity_info->maxFacePerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxCellPerItem() const
{
  return m_local_connectivity_info->maxCellPerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxLocalNodePerItemType() const
{
  return m_local_connectivity_info->maxNodeInItemTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxLocalEdgePerItemType() const
{
  return m_local_connectivity_info->maxEdgeInItemTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemFamily::
maxLocalFacePerItemType() const
{
  return m_local_connectivity_info->maxFaceInItemTypeInfo();
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
  return ItemVectorView(m_infos.itemsInternal(),local_ids);
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
addVariable(IVariable* var)
{
  //info() << "Add var=" << var->fullName() << " to family=" << name();
  if (var->itemFamily()!=this)
    ARCANE_FATAL("Can not add a variable to a different family");
  m_used_variables.insert(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeVariable(IVariable* var)
{
  //info() << "Remove var=" << var->fullName() << " to family=" << name();
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
  ARCANE_ASSERT((m_mesh->itemFamilyNetwork()),("Cannot call ItemFamily::removeItems2 if no ItemFamilyNetwork available."))
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
    ItemInternal* removed_item = m_infos.itemInternal(removed_item_lid);
    if (removed_item->isDetached()) {
      m_infos.removeDetachedOne(removed_item);
    }
    else {
      m_infos.removeOne(removed_item);
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
  //- Only cells are detached, ie no parent dependencies are to be found
  ARCANE_ASSERT((m_mesh->itemFamilyNetwork()->getParentDependencies(this).empty()),("Only cells are detached, no parent dependencies are to be found."))
  // Remove all the relations parent and child. Keep child dependencies => used when removing detach cells. No parent dependencies
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
    m_infos.detachOne(m_infos.itemInternal(detached_item_lid)); // when family/mesh endUpdate is done ? needed ?
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
removeNeedRemoveMarkedItems()
{
  if (!m_mesh->itemFamilyNetwork() || !IItemFamilyNetwork::plug_serializer)
    fatal() << "ItemFamily::removeNeedMarkedItems cannot be called if ItemFamilyNetwork is unplugged.";
  UniqueArray<ItemInternal*> items_to_remove;
  UniqueArray<Int32> items_to_remove_lids;
  items_to_remove.reserve(1000);
  items_to_remove_lids.reserve(1000);

  ItemInternalMap& item_map = itemsMap();
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,item_map){
    ItemInternal* item = nbid->value();
    Integer f = item->flags();
    if (f & ItemInternal::II_NeedRemove){
      f &= ~ItemInternal::II_NeedRemove & ItemInternal::II_Suppressed;
      item->setFlags(f);
      items_to_remove.add(item);
      items_to_remove_lids.add(item->localId());
    }
  }
  info() << "Number of " << itemKind() << " of family "<< name()<<" to remove: " << items_to_remove.size();
  if (items_to_remove.size() == 0)
    return;

  // Update connectivities => remove all con pointing on the removed items
  // Todo the nearly same procedure is done in _detachCells2 : mutualize in a method (watchout the connectivities used are parentConnectivities or parentRelations...)
  ItemScalarProperty<bool> is_removed_item;
  is_removed_item.resize(this,false);
  for (auto removed_item: items_to_remove) {
    is_removed_item[*removed_item] = true;
  }
//  for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentConnectivities(this)) {
  for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentRelations(this)) { // Should be getParentConnectivities, but because legacy connectivity cannot remove a connectivity with a Node as target, we need to restrain to Relations...
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
  m_infos.removeMany(items_to_remove_lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CompareUniqueIdWithSuppression
{
 public:
  bool operator()(const ItemInternal* item1,const ItemInternal* item2) const
  {
    // Il faut mettre les entités détruites en fin de liste
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

void ItemFamily::
addSourceConnectivity(IIncrementalItemConnectivity* c)
{
  m_source_incremental_item_connectivities.add(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
addTargetConnectivity(IIncrementalItemConnectivity* c)
{
  m_target_incremental_item_connectivities.add(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamily::
_checkValidConnectivity()
{
  {
    // Vérifie qu'il n'y a pas d'entité nulle.
    Integer index = 0;
    ENUMERATE_ITEM(i,allItems()){
      Item item = *i;
      if (item.null())
        ARCANE_FATAL("family={0}: local item lid={1} is null",fullName(),item);
      index++;
    }
  }
  for( Integer i=0; i<ItemInternalConnectivityList::MAX_ITEM_KIND; ++i ){
    Int32ConstArrayView con_list = m_item_connectivity_list.connectivityList(i);
    Int32ConstArrayView con_index = m_item_connectivity_list.connectivityIndex(i);
    info(4) << "Family name=" << fullName();
    info(4) << "I=" << i << " list_size=" << con_list.size()
            << " list_ptr=" << con_list.unguardedBasePointer()
            << " index_size=" << con_index.size();
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
 * \brief Positionne les infos Item::isShared() pour les entités de la
 * famille.
 *
 * Cette méthode n'est valide que si les infos de connectivité de
 * m_variable_synchronizer sont à jour.
 */
void ItemFamily::
_updateItemsSharedFlag()
{
  ItemInternalList items(_itemsInternal());
  for( Integer i=0, n=items.size(); i<n; ++i )
    items[i]->removeFlags(ItemInternal::II_Shared);
  Int32ConstArrayView comm_ranks = m_variable_synchronizer->communicatingRanks();
  Integer nb_rank = comm_ranks.size();
  // Parcours les sharedItems() du synchroniseur et positionne le flag
  // II_Shared pour les entités de la liste.
  for( Integer i=0; i<nb_rank; ++i ){
    Int32ConstArrayView shared_ids = m_variable_synchronizer->sharedItems(i);
    for( auto id : shared_ids )
      items[id]->addFlags(ItemInternal::II_Shared);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
