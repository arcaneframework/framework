// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMesh.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classe de gestion d'un maillage évolutif.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESH_H
#define ARCANE_MESH_DYNAMICMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/PerfCounterMng.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/List.h"

#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyModifier.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/IItemFamilyNetwork.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/MeshKind.h"

#include "arcane/mesh/SubMeshTools.h"
#include "arcane/mesh/MeshVariables.h"
#include "arcane/mesh/NewWithLegacyConnectivity.h"
#include "arcane/mesh/MeshEventsImpl.h"

#include <functional>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableMng;
class Properties;
class IUserDataList;
class IItemFamilyCompactPolicy;
class IMeshExchangeMng;
class IMeshCompactMng;
//! AMR
class IAMRTransportFunctor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshIncrementalBuilder;
class DynamicMeshChecker;
class ItemMemoryReferenceMng;
class MeshTiedInterface;
class TiedInterface;
class MeshPartitionConstraintMng;
class TiedInterfaceMng;
class NodeFamily;
class EdgeFamily;
class FaceFamily;
class CellFamily;
class DulaNodeFamily;
//! AMR
class MeshRefinement;
class NewItemOwnerBuilder;
class ExtraGhostCellsBuilder;
class ExtraGhostParticlesBuilder;

class DynamicMeshMergerHelper;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'un maillage.
 */
class ARCANE_MESH_EXPORT DynamicMesh
: public MeshVariables
, public TraceAccessor
, public IPrimaryMesh
, public IMeshModifier
, public IUnstructuredMeshInitialAllocator
, public ICartesianMeshInitialAllocator
{
  class InternalApi;

 private:
  // TEMPORAIRE
  friend DynamicMeshMergerHelper;
 public:

  typedef ItemInternal* ItemInternalPtr;
  typedef List<IItemFamily*> ItemFamilyList;

  class InitialAllocator
  : public IMeshInitialAllocator
  {
   public:

    InitialAllocator(DynamicMesh* m) : m_mesh(m){}
    IUnstructuredMeshInitialAllocator* unstructuredMeshAllocator() override
    {
      return m_mesh;
    }
    ICartesianMeshInitialAllocator* cartesianMeshAllocator() override
    {
      return m_mesh;
    }

   private:

    DynamicMesh* m_mesh;
  };

#ifdef ACTIVATE_PERF_COUNTER
  struct PerfCounter
  {
      typedef enum {
        UPGHOSTLAYER1,
        UPGHOSTLAYER2,
        UPGHOSTLAYER3,
        UPGHOSTLAYER4,
        UPGHOSTLAYER5,
        UPGHOSTLAYER6,
        UPGHOSTLAYER7,
        NbCounters
      }  eType ;

      static const std::string m_names[NbCounters] ;
  } ;
#endif
 public:

  void _checkKindRange(eItemKind ik) const // Il faut rendre fatal pour les particules, les dofs et le graph ??
  {
    if (ik==IK_Node || 
        ik==IK_Edge || 
        ik==IK_Face || 
        ik==IK_Cell ||
        ik==IK_DoF)
      return;
    throw ArgumentException(A_FUNCINFO,"Invalid Range");
  }
  
 public:
  
  DynamicMesh(ISubDomain* sd,const MeshBuildInfo& mbi, bool is_submesh);
  ~DynamicMesh();

 public:

  void build() override;

 public:
  
  MeshHandle handle() const override { return m_mesh_handle; }
  String name() const override { return m_name; }
  String factoryName() const override { return m_factory_name; }

  IMesh* mesh() override { return this; }
  Integer dimension() override { return m_mesh_dimension(); }
  void setDimension(Integer dim) override;
  VariableScalarInteger connectivity() override { return m_mesh_connectivity; }
  Integer nbNode() override;
  Integer nbEdge() override;
  Integer nbFace() override;
  Integer nbCell() override;
  
  Integer nbItem(eItemKind ik) override
  {
    _checkKindRange(ik);
    return m_item_families[ik]->nbItem();
  }
  
  ItemInternalArrayView itemsInternal(eItemKind ik) override
  {
    _checkKindRange(ik);
    return m_item_families[ik]->itemsInternal();
  }

  VariableNodeReal3& nodesCoordinates() override;
  SharedVariableNodeReal3 sharedNodesCoordinates() override;

 public:

  void setEstimatedCells(Integer nb_cell);

 public:

  void exchangeItems() override;
  void clearItems() override;

  /**
   * Met à jour les mailles fantômes
   *
   */
  void updateGhostLayers() override;
  void updateGhostLayers(bool remove_old_ghost) override;
  //! AMR
  void updateGhostLayerFromParent(Array<Int64>& ghost_cell_to_refine,
                                  Array<Int64>& cells_to_coarsen,
                                  bool remove_old_ghost) override;

  //! Fusionne les maillages de \a meshes avec le maillage actuel.
  void mergeMeshes(ConstArrayView<IMesh*> meshes) override;

  void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) override;
  void removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) override;
  void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) override;
  void removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) override;
  
  void serializeCells(ISerializer* buffer,Int32ConstArrayView cells_local_id) override;
  Int32 meshRank() { return m_mesh_part_info.partRank(); }

  void checkValidMesh() override;
  void checkValidMeshFull() override;
  void checkValidConnectivity();

 public:

  bool isAllocated() override { return m_is_allocated; }
   
  void reloadMesh() override;

  void deallocate() override;
  void allocateCells(Integer mesh_nb_cell,Int64ConstArrayView cells_info,bool one_alloc) override;
  void endAllocate() override;

  void addCells(Integer nb_cell,Int64ConstArrayView cell_infos,Int32ArrayView cells) override;
  void addCells(const MeshModifierAddCellsArgs& args) override;
  void addCells(ISerializer* buffer) override;
  void addCells(ISerializer* buffer,Int32Array& cells_local_id) override;
  void addFaces(Integer nb_face,Int64ConstArrayView face_infos,Int32ArrayView faces) override;
  void addFaces(const MeshModifierAddFacesArgs& args) override;
  void addEdges(Integer nb_edge,Int64ConstArrayView edge_infos,Int32ArrayView edges) override;
  void addNodes(Int64ConstArrayView nodes_uid,Int32ArrayView nodes) override;
  void removeCells(Int32ConstArrayView cells_local_id,bool update_graph) override;
  void removeCells(Int32ConstArrayView cells_local_id) override
  {
    removeCells(cells_local_id,true) ;
  }
  void detachCells(Int32ConstArrayView cells_local_id) override;
  void removeDetachedCells(Int32ConstArrayView cells_local_id) override;
  //! AMR
  void flagCellToRefine(Int32ConstArrayView cells_lids) override;
  void flagCellToCoarsen(Int32ConstArrayView cells_lids) override;
  void refineItems() override;
  void coarsenItems() override;
  void coarsenItemsV2(bool update_parent_flag) override;
  void compact() ;
  bool adapt() override;
  void registerCallBack(IAMRTransportFunctor* f) override;
  void unRegisterCallBack(IAMRTransportFunctor* f) override;
  void addHChildrenCells(Cell parent_cell,Integer nb_cell,
                         Int64ConstArrayView cells_infos,Int32ArrayView cells) override;

  void addParentCellToCell(Cell child, Cell parent) override;
  void addChildCellToCell(Cell parent, Cell child) override;

  void addParentFaceToFace(Face child, Face parent) override;
  void addChildFaceToFace(Face parent, Face child) override;

  void addParentNodeToNode(Node child, Node parent) override;
  void addChildNodeToNode(Node parent, Node child) override;

  void endUpdate() override;
  Int64 timestamp() override { return m_timestamp; }

  bool isPrimaryMesh() const override;
  IPrimaryMesh* toPrimaryMesh() override;

  //! Informations sur les parties du maillage
  const MeshPartInfo& meshPartInfo() const override { return m_mesh_part_info; }
  void setMeshPartInfo(const MeshPartInfo& mpi) override;
  IUserDataList* userDataList() override { return m_mesh_handle.meshUserDataList(); }
  const IUserDataList* userDataList() const override { return m_mesh_handle.meshUserDataList(); }

  IGhostLayerMng* ghostLayerMng() const override { return m_ghost_layer_mng; }
  IMeshUniqueIdMng* meshUniqueIdMng() const override { return m_mesh_unique_id_mng; }
  IMeshChecker* checker() const override;
  
public:

  VariableItemInt32& itemsNewOwner(eItemKind ik) override
  {
    _checkKindRange(ik);
    return m_item_families[ik]->itemsNewOwner();
  }

  ItemInternalMap& nodesMap();
  ItemInternalMap& edgesMap();
  ItemInternalMap& facesMap();
  ItemInternalMap& cellsMap();

  void addParentCells(ItemVectorView & items);

  // ItemFamilyNetwork
  void removeItems(IItemFamily* item_family, Int32ConstArrayView cells_local_id);

 private:

  ISubDomain* m_sub_domain;
  IMeshMng* m_mesh_mng;
  MeshHandle m_mesh_handle;
  IParallelMng* m_parallel_mng;
  MeshItemInternalList m_item_internal_list;
  IVariableMng* m_variable_mng;
  Properties* m_properties;
  Int64 m_timestamp;
  bool m_is_allocated;
  Integer m_dimension;
  String m_name;
  String m_factory_name;
  bool m_need_compact;
  bool m_has_edge;

 public:
  
  NodeFamily* m_node_family;
  EdgeFamily* m_edge_family;
  FaceFamily* m_face_family;
  CellFamily* m_cell_family;

 private:
  
  IMesh* m_parent_mesh;
  ItemGroupImpl* m_parent_group;
  UniqueArray<DynamicMesh*> m_child_meshes;

 public:

  NodeFamily& trueNodeFamily() { return *m_node_family; }
  EdgeFamily& trueEdgeFamily() { return *m_edge_family; }
  FaceFamily& trueFaceFamily() { return *m_face_family; }
  CellFamily& trueCellFamily() { return *m_cell_family; }
 
 public:

  NodeGroup allNodes() override { return allItems(IK_Node); }
  EdgeGroup allEdges() override { return allItems(IK_Edge); }
  FaceGroup allFaces() override { return allItems(IK_Face); }
  CellGroup allCells() override { return allItems(IK_Cell); }

  //TODO: supprimer cette méthode
  ItemGroup allItems(eItemKind kind)
  {
    _checkKindRange(kind);
    return m_item_families[kind]->allItems();
  }

  NodeGroup ownNodes() override { return ownItems(IK_Node); }
  EdgeGroup ownEdges() override { return ownItems(IK_Edge); }
  FaceGroup ownFaces() override { return ownItems(IK_Face); }
  CellGroup ownCells() override { return ownItems(IK_Cell); }

  //TODO: supprimer cette méthode
  ItemGroup ownItems(eItemKind kind)
  {
    _checkKindRange(kind);
    return m_item_families[kind]->allItems().own();
  }

  FaceGroup outerFaces() override;

  //! AMR
  //! Groupe de toutes les mailles actives
  CellGroup allActiveCells() override;

  //! Groupe de toutes les mailles actives et propres au domaine
  CellGroup ownActiveCells() override;

  //! Groupe de toutes les mailles de niveau \p level
  CellGroup allLevelCells(const Integer& level) override;

  //! Groupe de toutes les mailles propres de niveau \p level
  CellGroup ownLevelCells(const Integer& level) override;

  //! Groupe de toutes les faces actives
  FaceGroup allActiveFaces() override;

  //! Groupe de toutes les faces actives sur la frontière.
  FaceGroup ownActiveFaces() override;

  //! Groupe de toutes les faces actives
  FaceGroup innerActiveFaces() override;

  //! Groupe de toutes les faces actives sur la frontière.
  FaceGroup outerActiveFaces() override;

//  void readAmrActivator(const XmlNode& mesh_node) override;

 public:

  MeshItemInternalList* meshItemInternalList() override
  { return &m_item_internal_list; }

 public:

  ISubDomain* subDomain() override { return m_sub_domain; }
  IParallelMng* parallelMng() override;
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }

  ItemGroupCollection groups() override;

  void prepareForDump() override;
  void initializeVariables(const XmlNode& init_node) override;
  ItemGroup findGroup(const String& name) override;
  ItemGroup findGroup(const String& name,eItemKind ik,bool create_if_needed);
  void destroyGroups() override;

  NodeGroup findNodeGroup(const String& aname)
  { return findGroup(aname,IK_Node,false); }
  EdgeGroup findEdgeGroup(const String& aname)
  { return findGroup(aname,IK_Edge,false); }
  FaceGroup findFaceGroup(const String& aname)
  { return findGroup(aname,IK_Face,false); }
  CellGroup findCellGroup(const String& aname)
  { return findGroup(aname,IK_Cell,false); }

  ItemGroup createGroup(const String& aname,const ItemGroup& parent);
  ItemGroup createGroup(const String& aname,eItemKind ik);

  //! AMR
  bool  isAmrActivated() const override
  {
	  return m_is_amr_activated;
  }

 public:

  IItemFamily* createItemFamily(eItemKind ik,const String& name) override;

  IItemFamily* findItemFamily(eItemKind ik,const String& name,bool create_if_needed,bool register_modifier_if_created) override;
  IItemFamily* findItemFamily(const String& name,bool throw_exception=true) override;
  IItemFamilyModifier* findItemFamilyModifier(eItemKind ik,const String& name) override;
  void addItemFamilyModifier(IItemFamilyModifier*) ;

  IItemFamily* itemFamily(eItemKind ik) override
  {
    _checkKindRange(ik);
    return m_item_families[ik];
  }
  IItemFamily* nodeFamily() override;
  IItemFamily* edgeFamily() override;
  IItemFamily* faceFamily() override;
  IItemFamily* cellFamily() override;

  IItemFamilyCollection itemFamilies() override
  {
    return m_item_families;
  }
 
 public:
  
  bool isDynamic() const override
  {
    return m_is_dynamic;
  }

  void setDynamic(bool v) override
  {
    m_is_dynamic = v;
  }

  void setCheckLevel(Integer level) override;

  Integer checkLevel() const override;

 public:

  void computeTiedInterfaces(const XmlNode& xml_node) override;
  bool hasTiedInterface() override;
  TiedInterfaceCollection tiedInterfaces() override;
  IMeshPartitionConstraintMng* partitionConstraintMng() override;
  ConstArrayView<TiedInterface*> trueTiedInterfaces();

 public:

  IMeshUtilities* utilities() override;
  IMeshModifier* modifier() override { return this; }
  Properties* properties() override { return m_properties; }
  void synchronizeGroupsAndVariables() override;

 public:

  void defineParentForBuild(IMesh* mesh, ItemGroup group) override;
  IMesh* parentMesh() const override;
  ItemGroup parentGroup() const override;
  void addChildMesh(IMesh* sub_mesh) override;
  IMeshCollection childMeshes() const override;

  void setOwnersFromCells() override { _setOwnersFromCells(); }
  IMeshCompactMng* _compactMng() override { return m_mesh_compact_mng; }
  InternalConnectivityPolicy _connectivityPolicy() const override
  {
    return m_connectivity_policy;
  }

 public:

  DynamicMeshIncrementalBuilder* incrementalBuilder() { return m_mesh_builder; }
  MeshRefinement* meshRefinement() { return m_mesh_refinement; }
  void endUpdate(bool update_ghost_layer, bool remove_old_ghost) override;

 public:

  bool useMeshItemFamilyDependencies() const override  {return m_use_mesh_item_family_dependencies; }
  IItemFamilyNetwork* itemFamilyNetwork() override {return m_item_family_network;}
  IIndexedIncrementalItemConnectivityMng* indexedConnectivityMng() override { return m_indexed_connectivity_mng.get(); }

 public:

  IMeshMng* meshMng() const override { return m_mesh_mng; }
  IVariableMng* variableMng() const override { return m_variable_mng; }
  ItemTypeMng* itemTypeMng() const override { return m_item_type_mng; }

  void computeSynchronizeInfos() override;

 public:

  IMeshInitialAllocator* initialAllocator() override { return &m_initial_allocator; }
  void allocate(UnstructuredMeshAllocateBuildInfo& build_info) override;
  void allocate(CartesianMeshAllocateBuildInfo& build_info) override;

 public:

  EventObservable<const MeshEventArgs&>& eventObservable(eMeshEventType type) override
  {
    return m_mesh_events.eventObservable(type);
  }

 public:

  const MeshKind meshKind() const override { return m_mesh_kind; }

  IMeshInternal* _internalApi() override;
  IMeshModifierInternal* _modifierInternalApi() override;

 private:

  IMeshUtilities* m_mesh_utilities = nullptr;

 public:

  DynamicMeshIncrementalBuilder* m_mesh_builder = nullptr;

 private:

  DynamicMeshChecker* m_mesh_checker = nullptr;
  SubMeshTools* m_submesh_tools = nullptr;
  //! AMR
  MeshRefinement* m_mesh_refinement = nullptr;
  NewItemOwnerBuilder * m_new_item_owner_builder = nullptr;
  ExtraGhostCellsBuilder* m_extra_ghost_cells_builder = nullptr;
  ExtraGhostParticlesBuilder* m_extra_ghost_particles_builder = nullptr;
  InitialAllocator m_initial_allocator;
  std::unique_ptr<InternalApi> m_internal_api;

 private:
  
  //! AMR
  bool m_is_amr_activated = false;
  eMeshAMRKind m_amr_type;

  bool m_is_dynamic = false;

  //! Liste des groupes d'entités
  ItemGroupList m_all_groups;

  //! Liste des interfaces familles d'entités
  ItemFamilyList m_item_families;

  //! Liste des implémentations des familles d'entités
  UniqueArray<ItemFamily*> m_true_item_families;
  UniqueArray<IItemFamilyModifier*> m_family_modifiers; // used for item family network

  ObserverPool m_observer_pool;
  TiedInterfaceMng* m_tied_interface_mng = nullptr;
  bool m_is_sub_connectivity_set = false;
  bool m_tied_interface_need_prepare_dump = false;
  
  MeshPartitionConstraintMng* m_partition_constraint_mng = nullptr;
  IGhostLayerMng* m_ghost_layer_mng = nullptr;
  IMeshUniqueIdMng* m_mesh_unique_id_mng = nullptr;
  IMeshExchangeMng* m_mesh_exchange_mng = nullptr;
  IMeshCompactMng* m_mesh_compact_mng = nullptr;

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter> m_perf_counter;
#endif

  InternalConnectivityPolicy m_connectivity_policy;
  MeshPartInfo m_mesh_part_info;

  bool m_use_mesh_item_family_dependencies  = false ;
  IItemFamilyNetwork* m_item_family_network = nullptr;
  ItemTypeMng* m_item_type_mng = nullptr;
  std::unique_ptr<IIndexedIncrementalItemConnectivityMng> m_indexed_connectivity_mng;
  MeshKind m_mesh_kind;
  bool m_do_not_save_need_compact = false;

  MeshEventsImpl m_mesh_events;

 private:

  void _printMesh(std::ostream& ostr);
  void _allocateCells(Integer mesh_nb_cell,
                      Int64ConstArrayView cells_info,
                      Int32ArrayView cells = Int32ArrayView(),
                      bool allow_build_face = true); 

  const char* _className() const { return "Mesh"; }

  void _allocateCells2(DynamicMeshIncrementalBuilder* mib);
  void _itemsUniqueIdsToLocalIdsSorted(eItemKind item_kind,ArrayView<Integer> ids);
  void _prepareForDump();
  void _prepareForDumpReal();
  void _readFromDump();

  void _setOwnersFromCells();
  // Les méthodes _synchronizeXXX ne sont pas récursives sur les sous-maillages
  void _synchronizeGroupsAndVariables();
  void _synchronizeGroups();
  void _synchronizeVariables();
  void _writeMesh(const String& base_name);
  void _removeGhostItems() ;
  // AMR
  void _removeGhostChildItems();
  void _removeGhostChildItems2(Array<Int64>& cells_to_coarsen);
  void _checkAMR() const;

  void _sortInternalReferences();
  void _finalizeMeshChanged();
  void _compactItemInternalReferences();
  void _compactItems(bool do_sort,bool compact_variables_and_groups);
  void _checkValidItem(ItemInternal* item);

  void _computeSynchronizeInfos();
  void _computeFamilySynchronizeInfos();
  void _computeGroupSynchronizeInfos();
  void _exchangeItems(bool do_compact);
  void _exchangeItemsNew();
  void _checkDimension() const;
  void _checkConnectivity();
  void _writeCells(const String& filename);

  void _prepareTiedInterfacesForDump();
  void _readTiedInterfacesFromDump();
  void _applyTiedInterfaceStructuration(TiedInterface* tied_interface);
  void _deleteTiedInterfaces();

  void _multipleExchangeItems(Integer nb_exchange,Integer version,bool do_compact);
  void _addCells(ISerializer* buffer,Int32Array* cells_local_id);
  void _setSubConnectivity();
  void _setDimension(Integer dim);
  void _internalUpdateGhost(bool update_ghost_layer,bool remove_old_ghost);
  void _internalEndUpdateInit(bool update_sync_info);
  void _internalEndUpdateResizeVariables();
  void _internalEndUpdateFinal(bool print_stat);

  void _computeExtraGhostCells();
  void _computeExtraGhostParticles();
  
  void _notifyEndUpdateForFamilies();
  ItemFamily* _createNewFamily(eItemKind kind, const String & name);

  void _saveProperties();
  void _loadProperties();
  void _addFamily(ItemFamily* true_family);

  void _buildAndInitFamily(IItemFamily* family);
  IItemFamilyPolicyMng* _createFamilyPolicyMng(ItemFamily* family);
  void _applyCompactPolicy(const String& timer_name,
                           std::function<void(IItemFamilyCompactPolicy*)> functor);
  void _updateGroupsAfterRemove();
  void _printConnectivityPolicy();

  // Add a dependency (downward adjacencies only) between two family: ie the source family
  // is built on the target family (ex a cell is build owns its nodes)
  template <class SourceFamily, class TargetFamily>
  void _addDependency(SourceFamily* source_family, TargetFamily* target_family)
  {
    typedef typename NewWithLegacyConnectivityType<SourceFamily,TargetFamily>::type CType;
    String name = String::format("{0}{1}",source_family->name(),target_family->name());
    auto connectivity = new CType(source_family,target_family,name);
    m_item_family_network->addDependency(source_family,target_family,connectivity);
  }

  // Add a relation : source family "sees" target family (ex a face sees its cells). Often upward adjacencies.
  template <class SourceFamily, class TargetFamily>
  void _addRelation(SourceFamily* source_family, TargetFamily* target_family)
  {
    typedef typename NewWithLegacyConnectivityType<SourceFamily,TargetFamily>::type CType;
    String name = String::format("{0}{1}",source_family->name(),target_family->name());
    auto connectivity = new CType(source_family,target_family,name);
    m_item_family_network->addRelation(source_family,target_family,connectivity);
  }

  // Update family dependencies with set connectivities
  void _updateItemFamilyDependencies(VariableScalarInteger connectivity);

  // Serialize Item
  void _serializeItems(ISerializer* buffer,Int32ConstArrayView item_local_ids, IItemFamily* item_family);
  void _deserializeItems(ISerializer* buffer,Int32Array *item_local_ids, IItemFamily* item_family);
  void _fillSerializer(ISerializer* buffer, std::map<String, Int32UniqueArray>& serializedItems);

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
