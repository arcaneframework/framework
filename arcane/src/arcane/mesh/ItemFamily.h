// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamily.h                                                (C) 2000-2021 */
/*                                                                           */
/* Famille d'entités.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILY_H
#define ARCANE_MESH_ITEMFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/ObserverPool.h"

#include "arcane/mesh/DynamicMeshKindInfos.h"

#include "arcane/IItemConnectivity.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;
class IVariableSynchronizer;
class Properties;
class IItemConnectivity;
class IIncrementalItemConnectivity;
class IItemFamilyTopologyModifier;
class IMeshCompacter;
class ItemFamilyCompactInfos;
class ItemDataList;
template <typename T> class ItemScalarProperty;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemSharedInfoList;
class ItemConnectivityInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille d'entités.
 *
 * Cette classe est la classe de base des classes qui gèrent tout ce qui
 * est relatif à un ensemble d'entité de même genre, par exemple des noeuds.
 *
 * Pour chaque famille, on gère:
 * - la liste des entités, l'ajout (addItems()) ou la suppression (removeItems()).
 * - les groupes, leur création (createGroup()), leur recherche (findGroup()).
 * - la synchronisation des entités (synchronize()).
 */
class ARCANE_MESH_EXPORT ItemFamily
: public TraceAccessor
, public IItemFamily
{
 private:
  class Variables;
  class AdjencyInfo
  {
   public:
    AdjencyInfo(const ItemGroup& item_group,const ItemGroup& sub_item_group,
                eItemKind link_kind,Integer nb_layer)
    : m_item_group(item_group), m_sub_item_group(sub_item_group),
      m_link_kind(link_kind), m_nb_layer(nb_layer)
      {
      }
   public:
    ItemGroup m_item_group;
    ItemGroup m_sub_item_group;
    eItemKind m_link_kind;
    Integer m_nb_layer;
   public:
    bool operator<(const AdjencyInfo& rhs) const
    {
      if (m_item_group != rhs.m_item_group)
        return m_item_group < rhs.m_item_group;
      if (m_sub_item_group != rhs.m_sub_item_group)
        return m_sub_item_group < rhs.m_sub_item_group;
      if (m_link_kind != rhs.m_link_kind)
        return m_link_kind < rhs.m_link_kind;
      return m_nb_layer < rhs.m_nb_layer;
    }

  };
  typedef std::map<AdjencyInfo,ItemPairGroup> AdjencyGroupMap;
 public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;

 public:

  ItemFamily(IMesh* mesh,eItemKind ik,const String& name);
  ~ItemFamily() override; //<! Libère les ressources

 public:

  void build() override;

 public:

  const String& name() const override { return m_name; }
  const String& fullName() const override { return m_full_name; }
  eItemKind itemKind() const override { return m_infos.kind(); }
  Integer nbItem() const override;
  Int32 variableMaxSize() const override { return maxLocalId(); }
  Int32 maxLocalId() const override;
  ItemInternalList itemsInternal() override;
  VariableItemInt32& itemsNewOwner() override;
  IItemFamily * parentFamily() const override;
  void setParentFamily(IItemFamily * parent) override;
  Integer parentFamilyDepth() const override;
  void addChildFamily(IItemFamily * family) override;
  IItemFamilyCollection childFamilies() override;

  void checkValid() override;
  void checkValidConnectivity() override;
  void checkUniqueIds(Int64ConstArrayView unique_ids) override;

 public:

  ItemInternalMap& itemsMap() { return m_infos.itemsMap(); }

 public:

  void endUpdate() override;
  void partialEndUpdate() override;
  void partialEndUpdateGroup(const ItemGroup& group) override;
  void partialEndUpdateVariable(IVariable* variable) override;

 public:

  void itemsUniqueIdToLocalId(ArrayView<Int64> ids,bool do_fatal=true) const;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              Int64ConstArrayView unique_ids,
                              bool do_fatal) const override;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              ConstArrayView<ItemUniqueId> unique_ids,
                              bool do_fatal) const override;

 public:

  ISubDomain* subDomain() const override;
  ITraceMng* traceMng() const override;
  IMesh* mesh() const override;
  IParallelMng* parallelMng() const override;

 public:

  IItemConnectivityInfo* localConnectivityInfos() const override;
  IItemConnectivityInfo* globalConnectivityInfos() const override;

 public:

  void addItems(Int64ConstArrayView unique_ids,Int32ArrayView items) override;
  void addItems(Int64ConstArrayView unique_ids,ArrayView<Item> items) override;
  void addItems(Int64ConstArrayView unique_ids,ItemGroup item_group) override;
  void removeItems(Int32ConstArrayView local_ids,bool keep_ghost =false) override
  {
    internalRemoveItems(local_ids,keep_ghost);
  }
  void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost =false) override;
  void removeItems2(ItemDataList& item_data_list) override; // Remove items based on family dependencies (ItemFamilyNetwork)
  void removeNeedRemoveMarkedItems() override;
  void exchangeItems() override;
  ItemVectorView view(Int32ConstArrayView local_ids) override;
  ItemVectorView view() override;

  void mergeItems(Int32 local_id1,Int32 local_id2) override;
  Int32 getMergedItemLID(Int32 local_id1,Int32 local_id2) override;

  ItemInternal* findOneItem(Int64 uid) override { return m_infos.findOne(uid) ; }

 public:

  ItemGroup allItems() const override;

 public:

  void notifyItemsOwnerChanged() override;
  ItemGroup findGroup(const String& name) const override;
  ItemGroup findGroup(const String& name,bool create_if_needed) override;
  ItemGroup createGroup(const String& name,Int32ConstArrayView elements,bool do_override=false) override;
  ItemGroup createGroup(const String& name) override;
  ItemGroup createGroup(const String& name,const ItemGroup& parent,bool do_override=false) override;
  ItemGroupCollection groups() const override;
  void notifyItemsUniqueIdChanged() override;
  void resizeVariables(bool force_resize) override;
  void destroyGroups() override;

 public:

  IVariable* findVariable(const String& name,bool throw_exception) override;
  void usedVariables(VariableCollection collection) override;

 public:

  void prepareForDump() override;
  void readFromDump() override;
  void copyItemsValues(Int32ConstArrayView source,Int32ConstArrayView destination) override;
  void copyItemsMeanValues(Int32ConstArrayView first_source,
                           Int32ConstArrayView second_source,
                           Int32ConstArrayView destination) override;
  void compactItems(bool do_sort) override;
  virtual void compactReferences();

  void clearItems() override;

  const DynamicMeshKindInfos& infos() const { return m_infos; }
  Int64ArrayView* uniqueIds();

 public:

  void setHasUniqueIdMap(bool v) override;
  bool hasUniqueIdMap() const override;

 public:

  void computeSynchronizeInfos() override;
  void getCommunicatingSubDomains(Int32Array& sub_domains) const override;
  void synchronize(VariableCollection variables) override;
  IVariableSynchronizer* allItemsSynchronizer() override;
  
  void reduceFromGhostItems(IVariable* v,IDataOperation* operation) override;
  void reduceFromGhostItems(IVariable* v,Parallel::eReduceType operation) override;

 public:
  
  GroupIndexTable* localIdToIndex(ItemGroup group);
 
 public:

  ItemPairGroup findAdjencyItems(const ItemGroup& group,
                                 const ItemGroup& sub_group,eItemKind link_kind,
                                 Integer layer) override;
  IParticleFamily* toParticleFamily() override { return nullptr; }
  void setItemSortFunction(IItemInternalSortFunction* sort_function) override;
  IItemInternalSortFunction* itemSortFunction() const override;
  void endAllocate() override;

 public:
  
  void addVariable(IVariable* var) override;
  void removeVariable(IVariable* var) override;

 public:
  
  void notifyEndUpdateFromMesh() override;

  void addSourceConnectivity(IItemConnectivity* connectivity) override;
  void addSourceConnectivity(IIncrementalItemConnectivity* c) override;
  void addTargetConnectivity(IItemConnectivity* connectivity) override;
  void addTargetConnectivity(IIncrementalItemConnectivity* c) override;
  void removeSourceConnectivity(IItemConnectivity* connectivity) override;
  void removeTargetConnectivity(IItemConnectivity* connectivity) override;
  void setConnectivityMng(IItemConnectivityMng* connectivity_mng) override;

  void addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items, Int32ConstArrayView owners) override;
  IItemFamilyPolicyMng* policyMng() override { return m_policy_mng; }
  Properties* properties() override { return m_properties; }
  IItemFamilyTopologyModifier* _topologyModifier() override { return m_topology_modifier; }
  ItemInternalConnectivityList* _unstructuredItemInternalConnectivityList() override
  {
    return itemInternalConnectivityList();
  }

 public:

  //NOTE: Cette méthode n'est pas virtuelle et seul pour l'instant DynamicMesh peut modifier la politique.
  void setPolicyMng(IItemFamilyPolicyMng* policy_mng);

 public:

  void beginCompactItems(ItemFamilyCompactInfos& compact_infos);
  /*! Compactage effectif des variables, groupes et familles enfant
   *  à partir des données du DynamicMeshKindInfos */
  void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos);
  void finishCompactItems(ItemFamilyCompactInfos& compact_infos);
  void removeItem(ItemInternal* item)
  {
    _removeOne(item);
  }
  //! Accesseur pour les connectités via Item et ItemInternal
  ItemInternalConnectivityList* itemInternalConnectivityList()
  {
    return &m_item_connectivity_list;
  }

 protected:

  void _removeOne(ItemInternal* item)
  {
    // TODO: vérifier en mode check avec les nouvelles connectivités que l'entité supprimée
    // n'a pas d'objets connectés.
    m_infos.removeOne(item);
  }
  void _detachOne(ItemInternal* item)
  {
    m_infos.detachOne(item);
  }
  ItemInternalList _itemsInternal()
  {
    return m_infos.itemsInternal();
  }
  ItemInternal* _itemInternal(Int32 local_id)
  {
    return m_infos.itemInternal(local_id);
  }
  ItemInternal* _allocOne(Int64 unique_id)
  {
    return m_infos.allocOne(unique_id);
  }
  ItemInternal* _allocOne(Int64 unique_id,bool& need_alloc)
  {
    return m_infos.allocOne(unique_id,need_alloc);
  }
  ItemInternal* _findOrAllocOne(Int64 uid,bool& is_alloc)
  {
    return m_infos.findOrAllocOne(uid,is_alloc);
  }
  void _setHasUniqueIdMap(bool v)
  {
    m_infos.setHasUniqueIdMap(v);
  }
  void _removeMany(Int32ConstArrayView local_ids)
  {
    m_infos.removeMany(local_ids);
  }
  void _removeDetachedOne(ItemInternal* item)
  {
    m_infos.removeDetachedOne(item);
  }

  void _detachCells2(Int32ConstArrayView local_ids);

 private:

 protected:

  String m_name;
  String m_full_name;
  IMesh* m_mesh;
  ISubDomain* m_sub_domain;
  IItemFamily * m_parent_family;
  Integer m_parent_family_depth;
 private:
  DynamicMeshKindInfos m_infos;
 protected:
  ItemGroupList m_item_groups;
  bool m_need_prepare_dump;
  MeshItemInternalList* m_item_internal_list;
  ItemSharedInfoList* m_item_shared_infos;
  ObserverPool m_observers;
  Ref<IVariableSynchronizer> m_variable_synchronizer;
  Integer m_current_variable_item_size;
  IItemInternalSortFunction* m_item_sort_function;
  std::set<IVariable*> m_used_variables;
  UniqueArray<ItemFamily*> m_child_families;
  ItemConnectivityInfo* m_local_connectivity_info;
  ItemConnectivityInfo* m_global_connectivity_info;
  Properties* m_properties;
  typedef std::set<IItemConnectivity*> ItemConnectivitySet;
  ItemConnectivitySet m_source_item_connectivities; //! connectivite ou ItemFamily == SourceFamily
  ItemConnectivitySet m_target_item_connectivities;   //! connectivite ou ItemFamily == TargetFamily
  IItemConnectivityMng* m_connectivity_mng;
  UniqueArray<IIncrementalItemConnectivity*> m_source_incremental_item_connectivities;
  UniqueArray<IIncrementalItemConnectivity*> m_target_incremental_item_connectivities;
  IItemFamilyPolicyMng* m_policy_mng;

 protected:

  void _checkNeedEndUpdate() const;
  void _updateSharedInfoAdded(ItemInternal* item);
  void _updateSharedInfoRemoved4(ItemInternal* item);
  void _updateSharedInfoRemoved7(ItemInternal* item);

  void _allocateInfos(ItemInternal* item,Int64 uid,ItemSharedInfo* isi);
  void _allocateInfos(ItemInternal* item,Int64 uid,ItemTypeInfo* type);
  void _endUpdate(bool need_check_remove);
  bool _partialEndUpdate();
  //void _partialEndUpdateGroup(const ItemGroup& group);
  void _updateGroup(ItemGroup group,bool need_check_remove);
  void _updateVariable(IVariable* var);
  //void _partialEndUpdateVariable(IVariable* variable);

  void _addConnectivitySelector(ItemConnectivitySelector* selector);
  void _buildConnectivitySelectors();

 private:

  Int32Array* m_items_data = nullptr;
  Int64Array* m_items_unique_id = nullptr;
  Int32Array* m_items_owner = nullptr;
  Int32Array* m_items_flags = nullptr;
  Int16Array* m_items_typeid = nullptr;
  Int64ArrayView m_items_unique_id_view;
  Variables* m_internal_variables = nullptr;
  Int32 m_default_sub_domain_owner;
 protected:
  Int32 m_sub_domain_id;
 private:
  bool m_is_parallel;
  /*! \brief Identifiant de la famille.
   *
   * Cet identifiant est incrémenté à chaque fois que la famille change.
   * Il est sauvegardé lors d'une protection et en cas de relecture, par
   * exemple suite à un retour-arrière. Si cet identifiant est le même
   * que celui sauvegardé, cela signifie que la famille n'a pas changée
   * depuis la dernière sauvegarde et donc qu'il n'y a aucune recréation
   * d'entité à faire.
   */
  Integer m_current_id;
  bool m_item_need_prepare_dump;
 private:
  Int64 m_nb_allocate_info;
 private:

  AdjencyGroupMap m_adjency_groups;
  UniqueArray<ItemConnectivitySelector*> m_connectivity_selector_list;
  IItemFamilyTopologyModifier* m_topology_modifier;
  //! Accesseur pour les connectités via Item et ItemInternal
  ItemInternalConnectivityList m_item_connectivity_list;

  UniqueArray<ItemConnectivitySelector*> m_connectivity_selector_list_by_item_kind;
  bool m_use_legacy_compact_item = false;

 protected:

  virtual IItemInternalSortFunction* _defaultItemSortFunction();
  void _reserveInfosMemory(Integer memory);
  void _resizeInfos(Integer memory);

  ItemSharedInfo* _findSharedInfo(ItemTypeInfo* type);

  Integer _allocMany(Integer memory);
  void _setSharedInfosPtr(Integer* ptr);
  void _copyInfos(ItemInternal* item,ItemSharedInfo* old_isi,ItemSharedInfo* new_isi);
  void _checkValid();
  void _checkValidConnectivity();
  void _notifyDataIndexChanged();
  void _processNewGroup(ItemGroup group);
  String _variableName(const String& base_name) const;
  template<class Type> void
  _synchronizeVariable(IVariable* var,Type* var_value,Integer nb_elem);
  void _updateGroups(bool check_need_remove);
  void _compactFromParentFamily(const ItemFamilyCompactInfos& compact_infos);
  void _checkComputeSynchronizeInfos(Int32 changed);
  void _readGroups();
  void _invalidateComputedGroups();
  void _compactItems(bool do_sort);
  void _compactOnlyItems(bool do_sort);
  void _applyCheckNeedUpdateOnGroups();
  void _setTopologyModifier(IItemFamilyTopologyModifier* tm);

 private:
  
  inline void _setUniqueId(Int32 lid,Int64 uid);
  void _setSharedInfosBasePtr();
  void _setSharedInfosNoCopy(ItemInternal* item,ItemSharedInfo* isi);
  void _setDataIndexForItem(ItemInternal* item,Int32 data_index);
  void _setSharedInfoForItem(ItemInternal* item,ItemSharedInfo* isi,Int32 data_index);
  void _updateItemsSharedFlag();

 protected:

  void _checkValidItem(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    arcaneThrowIfNull(item,"item","Invalid null item");
#else
    ARCANE_UNUSED(item);
#endif
  }
  void _checkValidSourceTargetItems(ItemInternal* source,ItemInternal* target)
  {
#ifdef ARCANE_CHECK
    arcaneThrowIfNull(source,"source","Invalid null source item");
    arcaneThrowIfNull(target,"target","Invalid null target item");
#else
    ARCANE_UNUSED(source);
    ARCANE_UNUSED(target);
#endif
  }

 private:

  void _getConnectedItems(IIncrementalItemConnectivity* parent_connectivity,ItemVector& target_family_connected_items);
  void _fillHasExtraParentProperty(ItemScalarProperty<bool>& child_families_has_extra_parent,ItemVectorView connected_items);
  void _computeConnectivityInfo(ItemConnectivityInfo* ici);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ARcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
