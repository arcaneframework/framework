// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamily.h                                                (C) 2000-2025 */
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

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/IItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemGroup.h"

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
class ItemInternalMap;
class DynamicMeshKindInfos;
class ItemSharedInfoList;
class ItemConnectivityInfo;
class ItemConnectivitySelector;

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
  class InternalApi;
  class AdjacencyInfo
  {
   public:

    AdjacencyInfo(const ItemGroup& item_group, const ItemGroup& sub_item_group,
                  eItemKind link_kind, Integer nb_layer)
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

    bool operator<(const AdjacencyInfo& rhs) const
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
  using AdjacencyGroupMap = std::map<AdjacencyInfo, ItemPairGroup>;
  // Garde la version avec faute d'ortographe pour des raisons de compatibilité
  // TODO: à enlever mi 2025
  using AdjencyGroupMap = AdjacencyGroupMap;

 public:

  using ItemInternalMap = ::Arcane::mesh::ItemInternalMap;

 public:

  ItemFamily(IMesh* mesh,eItemKind ik,const String& name);
  ~ItemFamily() override; //<! Libère les ressources

 public:

  void build() override;

 public:

  String name() const override { return m_name; }
  String fullName() const override { return m_full_name; }
  eItemKind itemKind() const override;
  Integer nbItem() const override;
  Int32 maxLocalId() const override;
  ItemInternalList itemsInternal() override;
  ItemInfoListView itemInfoListView() override;
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

  ItemInternalMap& itemsMap();

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

  void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost =false) override;
  void removeItems2(ItemDataList& item_data_list) override; // Remove items based on family dependencies (ItemFamilyNetwork)
  void removeNeedRemoveMarkedItems() override;

  ItemVectorView view(Int32ConstArrayView local_ids) override;
  ItemVectorView view() override;

  ItemInternal* findOneItem(Int64 uid) override;

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
  void clearItems() override;

  Int64ArrayView* uniqueIds();

  ItemSharedInfo* commonItemSharedInfo() { return m_common_item_shared_info; }

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Use _infos() instead.")
  const DynamicMeshKindInfos& infos() const;

 public:

  void setHasUniqueIdMap(bool v) override;
  bool hasUniqueIdMap() const override;

 public:

  void computeSynchronizeInfos() override;
  void getCommunicatingSubDomains(Int32Array& sub_domains) const override;
  void synchronize(VariableCollection variables) override;
  void synchronize(VariableCollection variables, Int32ConstArrayView local_ids) override;
  IVariableSynchronizer* allItemsSynchronizer() override;
  
  void reduceFromGhostItems(IVariable* v,IDataOperation* operation) override;
  void reduceFromGhostItems(IVariable* v,Parallel::eReduceType operation) override;

 public:

  ARCANE_DEPRECATED_REASON("Y2024: use findAdjacencyItems() instead")
  ItemPairGroup findAdjencyItems(const ItemGroup& group,
                                 const ItemGroup& sub_group,eItemKind link_kind,
                                 Integer layer) override;
  ItemPairGroup findAdjacencyItems(const ItemGroup& group,
                                   const ItemGroup& sub_group, eItemKind link_kind,
                                   Integer layer) override;
  IParticleFamily* toParticleFamily() override { return nullptr; }
  void setItemSortFunction(IItemInternalSortFunction* sort_function) override;
  IItemInternalSortFunction* itemSortFunction() const override;

 public:

  void addSourceConnectivity(IItemConnectivity* connectivity) override;
  void addTargetConnectivity(IItemConnectivity* connectivity) override;
  void removeSourceConnectivity(IItemConnectivity* connectivity) override;
  void removeTargetConnectivity(IItemConnectivity* connectivity) override;
  void setConnectivityMng(IItemConnectivityMng* connectivity_mng) override;

  void addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items, Int32ConstArrayView owners) override;
  EventObservableView<const ItemFamilyItemListChangedEventArgs&> itemListChangedEvent() override;

  IItemFamilyPolicyMng* policyMng() override { return m_policy_mng; }
  Properties* properties() override { return m_properties; }
  IItemFamilyInternal* _internalApi() override;

 public:

  //NOTE: Cette méthode doit être virtuelle pour que PolyhedralMesh puisse positionner la politique.
  virtual void setPolicyMng(IItemFamilyPolicyMng* policy_mng);

 public:

  void beginCompactItems(ItemFamilyCompactInfos& compact_infos);
  /*! Compactage effectif des variables, groupes et familles enfant
   *  à partir des données du DynamicMeshKindInfos */
  void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos);
  void finishCompactItems(ItemFamilyCompactInfos& compact_infos);
  void removeItem(Item item)
  {
    _removeOne(item);
  }
  //! Accesseur pour les connectivités via Item et ItemInternal
  ItemInternalConnectivityList* itemInternalConnectivityList()
  {
    return &m_item_connectivity_list;
  }

 protected:

  void _removeOne(Item item);
  void _detachOne(Item item);
  ItemInternalList _itemsInternal();
  ItemInternal* _itemInternal(Int32 local_id);
  ItemInternal* _allocOne(Int64 unique_id);
  ItemInternal* _allocOne(Int64 unique_id, bool& need_alloc);
  ItemInternal* _findOrAllocOne(Int64 uid, bool& is_alloc);
  void _setHasUniqueIdMap(bool v);
  void _removeMany(Int32ConstArrayView local_ids);
  void _removeDetachedOne(Item item);
  const DynamicMeshKindInfos& _infos() const;

  void _detachCells2(Int32ConstArrayView local_ids);

  virtual void _endAllocate();
  virtual void _notifyEndUpdateFromMesh();

 protected:

  String m_name;
  String m_full_name;
  IMesh* m_mesh = nullptr;
  InternalApi* m_internal_api = nullptr;
  ISubDomain* m_sub_domain = nullptr;
  IItemFamily* m_parent_family = nullptr;
  Integer m_parent_family_depth = 0;

 private:

  std::unique_ptr<DynamicMeshKindInfos> m_infos;

 protected:

  ItemGroupList m_item_groups;
  bool m_need_prepare_dump = true;
  MeshItemInternalList* m_item_internal_list = nullptr;

 private:

  ItemSharedInfo* m_common_item_shared_info = nullptr;

 protected:

  ItemSharedInfoList* m_item_shared_infos = nullptr;
  ObserverPool m_observers;
  Ref<IVariableSynchronizer> m_variable_synchronizer;
  Integer m_current_variable_item_size = 0;
  IItemInternalSortFunction* m_item_sort_function = nullptr;
  std::set<IVariable*> m_used_variables;
  UniqueArray<ItemFamily*> m_child_families;
  ItemConnectivityInfo* m_local_connectivity_info = nullptr;
  ItemConnectivityInfo* m_global_connectivity_info = nullptr;
  Properties* m_properties = nullptr;
  typedef std::set<IItemConnectivity*> ItemConnectivitySet;
  ItemConnectivitySet m_source_item_connectivities; //! connectivite ou ItemFamily == SourceFamily
  ItemConnectivitySet m_target_item_connectivities;   //! connectivite ou ItemFamily == TargetFamily
  IItemConnectivityMng* m_connectivity_mng = nullptr;

 private:

  UniqueArray<Ref<IIncrementalItemSourceConnectivity>> m_source_incremental_item_connectivities;
  UniqueArray<Ref<IIncrementalItemTargetConnectivity>> m_target_incremental_item_connectivities;

 protected:

  IItemFamilyPolicyMng* m_policy_mng = nullptr;

 protected:

  void _checkNeedEndUpdate() const;
  void _updateSharedInfo();

  void _allocateInfos(ItemInternal* item,Int64 uid,ItemSharedInfoWithType* isi);
  void _allocateInfos(ItemInternal* item,Int64 uid,ItemTypeInfo* type);
  void _endUpdate(bool need_check_remove);
  bool _partialEndUpdate();
  void _updateGroup(ItemGroup group,bool need_check_remove);
  void _updateVariable(IVariable* var);

  void _addConnectivitySelector(ItemConnectivitySelector* selector);
  void _buildConnectivitySelectors();
  void _preAllocate(Int32 nb_item,bool pre_alloc_connectivity);
  ItemInternalConnectivityList* _unstructuredItemInternalConnectivityList()
  {
    return itemInternalConnectivityList();
  }

 public:

  IItemFamilyTopologyModifier* _topologyModifier() override { return m_topology_modifier; }
  void resizeVariables(bool force_resize) override { _resizeVariables(force_resize); }

 private:

  Int64Array* m_items_unique_id = nullptr;
  Int32Array* m_items_owner = nullptr;
  Int32Array* m_items_flags = nullptr;
  Int16Array* m_items_type_id = nullptr;
  Int32Array* m_items_nb_parent = nullptr;
  // TODO: a supprimer car redondant avec le champ correspondant de ItemSharedInfo
  Int64ArrayView m_items_unique_id_view;
  Variables* m_internal_variables = nullptr;
  Int32 m_default_sub_domain_owner = A_NULL_RANK;

 protected:

  Int32 m_sub_domain_id = A_NULL_RANK;

 private:

  bool m_is_parallel = false;

  /*!
   * \brief Identifiant de la famille.
   *
   * Cet identifiant est incrémenté à chaque fois que la famille change.
   * Il est sauvegardé lors d'une protection et en cas de relecture, par
   * exemple suite à un retour-arrière. Si cet identifiant est le même
   * que celui sauvegardé, cela signifie que la famille n'a pas changée
   * depuis la dernière sauvegarde et donc qu'il n'y a aucune recréation
   * d'entité à faire.
   */
  Integer m_current_id  = 0;

  bool m_item_need_prepare_dump = false;

 private:

  Int64 m_nb_allocate_info = 0;

 private:

  AdjacencyGroupMap m_adjacency_groups;
  UniqueArray<ItemConnectivitySelector*> m_connectivity_selector_list;
  IItemFamilyTopologyModifier* m_topology_modifier = nullptr;
  //! Accesseur pour les connectivités via Item et ItemInternal
  ItemInternalConnectivityList m_item_connectivity_list;

  UniqueArray<ItemConnectivitySelector*> m_connectivity_selector_list_by_item_kind;
  bool m_use_legacy_compact_item = false;

 private:

  ItemTypeMng* m_item_type_mng = nullptr;
  bool m_do_shrink_after_allocate = false;

 protected:

  ItemTypeMng* _itemTypeMng() const { return m_item_type_mng; }
  virtual IItemInternalSortFunction* _defaultItemSortFunction();

  ARCANE_DEPRECATED_REASON("Y2022: This method is a now a no-operation")
  void _reserveInfosMemory(Integer memory);
  ARCANE_DEPRECATED_REASON("Y2022: This method is a now a no-operation")
  void _resizeInfos(Integer memory);

  ItemSharedInfoWithType* _findSharedInfo(ItemTypeInfo* type);

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer _allocMany(Integer memory);
  void _setSharedInfosPtr(Integer* ptr);
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
  void _checkValidItem(Item item) { _checkValidItem(ItemCompatibility::_itemInternal(item)); }
  void _checkValidSourceTargetItems(Item source,Item target)
  {
    _checkValidItem(ItemCompatibility::_itemInternal(source));
    _checkValidItem(ItemCompatibility::_itemInternal(target));
  }

 private:

  void _getConnectedItems(IIncrementalItemConnectivity* parent_connectivity,ItemVector& target_family_connected_items);
  void _fillHasExtraParentProperty(ItemScalarProperty<bool>& child_families_has_extra_parent,ItemVectorView connected_items);
  void _computeConnectivityInfo(ItemConnectivityInfo* ici);
  void _updateItemViews();
  void _resizeItemVariables(Int32 new_size,bool force_resize);
  void _handleOldCheckpoint();

  void _addSourceConnectivity(IIncrementalItemSourceConnectivity* c);
  void _addTargetConnectivity(IIncrementalItemTargetConnectivity* c);
  void _addVariable(IVariable* var);
  void _removeVariable(IVariable* var);
  void _resizeVariables(bool force_resize);
  void _shrinkConnectivityAndPrintInfos();
  void _addOnSizeChangedObservable(VariableRef& var_ref);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ARcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
