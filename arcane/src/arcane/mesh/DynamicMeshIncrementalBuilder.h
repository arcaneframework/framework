// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshIncrementalBuilder.h                             (C) 2000-2025 */
/*                                                                           */
/* Construction d'un maillage de manière incrémentale.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHINCREMENTALBUILDER_H
#define ARCANE_MESH_DYNAMICMESHINCREMENTALBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamilyNetwork.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemInternal.h"

#include "arcane/mesh/DynamicMeshKindInfos.h"
#include "arcane/mesh/ItemData.h"
#include "arcane/mesh/FullItemInfo.h"

#include <list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class SerializeBuffer;
}

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class OneMeshItemAdder;
class GhostLayerBuilder;
class FaceUniqueIdBuilder;
class EdgeUniqueIdBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction d'un maillage de manière incrémentale.
 */
class DynamicMeshIncrementalBuilder
: public TraceAccessor
{
 public:

  //! Construit une instance pour le maillage \a mesh
  explicit DynamicMeshIncrementalBuilder(DynamicMesh* mesh);
  ~DynamicMeshIncrementalBuilder();

 public:

  // Ajout de liste d'items du maillage et graphe
  
  void addCells(Integer nb_cell,Int64ConstArrayView cell_infos,
                Integer sub_domain_id,Int32ArrayView cells,
                bool allow_build_face = true);
  void addCells2(Integer nb_cell,Int64ConstArrayView cell_infos,
                 Integer sub_domain_id,Int32ArrayView cells,
                 bool allow_build_face = true);
  void addCells3(Integer nb_cell,Int64ConstArrayView cell_infos,
                   Integer sub_domain_id,Int32ArrayView cells,
                   bool allow_build_face = true);
  void addFaces(Integer nb_face,Int64ConstArrayView face_infos,
                Integer sub_domain_id,Int32ArrayView faces);
  void addFaces2(Integer nb_face,Int64ConstArrayView face_infos,
                  Integer sub_domain_id,Int32ArrayView faces);
  void addFaces3(Integer nb_face,Int64ConstArrayView face_infos,
                    Integer sub_domain_id,Int32ArrayView faces);
  void addEdges(Integer nb_edge,Int64ConstArrayView edge_infos,
                Integer sub_domain_id,Int32ArrayView edges);
  void addEdges2(Integer nb_edge,Int64ConstArrayView edge_infos,
                  Integer sub_domain_id,Int32ArrayView edges);
  void addEdges3(Integer nb_edge,Int64ConstArrayView edge_infos,
                    Integer sub_domain_id,Int32ArrayView edges);
  void addNodes(Int64ConstArrayView nodes_uid,
                Integer sub_domain_id,Int32ArrayView nodes);
  void addNodes2(Int64ConstArrayView nodes_uid,
                Integer sub_domain_id,Int32ArrayView nodes);
  
  void addItems(ItemDataList& item_info_list);
  void addItems(ItemDataList& item_info_list, ItemDataList& item_relation_info_list);
  void addRelations(ItemDataList& item_relation_list);

  void addFamilyItems(ItemData& item_info);

  //! Ajout au maillage courant d'item venant d'un maillage parent
  void addParentCells(const ItemVectorView & items);
  //! Ajout au maillage courant d'item venant d'un maillage parent
  void addParentItems(const ItemVectorView & items, const eItemKind submesh_kind);
  //! AMR
  //! Ajout au maillage courant des mailles enfants de la maille mère \p hParent_cell
  void addHChildrenCells(Cell hParent_cell,Integer nb_cell,Int64ConstArrayView cells_infos,
           Int32 sub_domain_id,Int32ArrayView cells,
           bool allow_build_face);
  void computeFacesUniqueIds();
  void addGhostLayers(bool is_allocate);
  //! AMR
  void addGhostChildFromParent(Array<Int64>& ghost_cell_to_refine);

  void removeGhostCells();
  void removeNeedRemoveMarkedCells();
 
 public:

  void removeNeedRemoveMarkedItems();

  void readFromDump();

  //! Définit la connectivité active pour le maillage associé
  /*! Ceci conditionne les connectivités à la charge de cette famille */
  void setConnectivity(Integer c);

  //! Remise à zéro des structures pour pouvoir faire à nouveau une allocation
  void resetAfterDeallocate();

 public:

  void printInfos();
  void printStats(Int32 level = TraceMessage::DEFAULT_LEVEL);
  
 public:
  
  ItemInternalMap& itemsMap(eItemKind ik);
  DynamicMesh* mesh() { return m_mesh; }
  bool isVerbose() const { return m_verbose; }
  bool hasEdge() const { return m_has_edge; }

  OneMeshItemAdder* oneMeshItemAdder() const { return m_one_mesh_item_adder; }

 private:

  struct NodeInFace;

  struct NodeInFacePtr
  {
    NodeInFacePtr(const Int64& node_uid) : m_ptr(std::make_shared<NodeInFace>(node_uid)){}
    bool operator< (const NodeInFacePtr& a) const {return (*m_ptr) < (*a.m_ptr);}
    NodeInFace* operator->() {return m_ptr.operator->();}
    const NodeInFace* operator->() const {return m_ptr.operator->();}
    bool operator==(const NodeInFacePtr& a) const {return (*m_ptr) == (*a.m_ptr);}
    std::shared_ptr<NodeInFace> m_ptr;
  };

  struct NodeInFace
  {
//    typedef std::shared_ptr<NodeInFace> NodeInFacePtr;
//    typedef std::set<NodeInFacePtr, std::function<bool(const NodeInFacePtr&, const NodeInFacePtr&)>> NodeInFaceSet; // Does not work bad function call exception (gcc 4.7.2 pb ?). Need to define NodeInFacePtr class
//    typedef std::set<NodeInFacePtr> NodeInFaceSet;
    typedef std::list<NodeInFacePtr> NodeInFaceSet;

    NodeInFace(const Int64& node_uid)
      : m_uid(node_uid)
      , m_face_uid(NULL_ITEM_ID){}
//      , m_next_node_set([](const NodeInFacePtr& a, const NodeInFacePtr& b){return (*a) < (*b);}){} // Does not work (gcc 4.7.2 pb ?). Need to define NodeInFacePtr class

    friend bool operator<(const NodeInFace& a,const NodeInFace & nif)
    {
      return a.m_uid < nif.m_uid;
    }
    friend bool operator==(const NodeInFace& a,const NodeInFace& b)
    {
      return a.m_uid == b.m_uid;
    }

    Int64 faceUid() const {return m_face_uid;}
    void setFaceUid(Int64 face_uid) {m_face_uid = face_uid;}

    const NodeInFaceSet& nextNodeSet() const {return m_next_node_set;}
    NodeInFaceSet& nextNodeSet() {return m_next_node_set;}
//    void setNextNode(const NodeInFacePtr & next_node) {m_next_node_set.push_back(next_node);}

    void print(){
      std::cout << "Node " << m_uid << " has set " << & m_next_node_set << " containing nodes : " << std:: endl;
      for (auto node : m_next_node_set)
        {
          node->print();
        }
    }

  public://for DEBUG
//  private:
    Int64 m_uid;
    Int64 m_face_uid;
    NodeInFaceSet m_next_node_set;
  };
//  typedef NodeInFace::NodeInFacePtr NodeInFacePtr;
  typedef NodeInFace::NodeInFaceSet NodeInFaceSet;

 private:
  
  void _printCellFaceInfos(ItemInternal* cell,const String& str);

  void _removeNeedRemoveMarkedItems(ItemInternalMap& map, UniqueArray<Int32>& items_local_id);

  void _fillFaceInfo(Integer& nb_face, Integer nb_cell,Int64Array& faces_infos, Int64ConstArrayView cells_infos, std::map<Int64,Int64SharedArray>& cell_to_face_connectivity_info);
  void _fillEdgeInfo(Integer& nb_edge, Integer nb_face,Int64Array& edges_infos, Int64ConstArrayView faces_infos, std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map);
  void _fillNodeInfo(Integer& nb_node, Integer nb_face,Int64Array& nodes_infos, Int64ConstArrayView faces_infos);
  void _fillNodeInfoFromEdge(Integer& nb_node, Integer nb_edge,Int64Array& nodes_infos, Int64ConstArrayView edges_infos);

  void _fillCellInfo2(Integer nb_cell,Int64ConstArrayView cells_infos,Int64Array& cell_infos2, Integer& nb_face, Int64Array& faces_infos, Int64Array& node_uids, bool allow_build_face);
  void _fillFaceInfo2(Integer nb_face,Int64ConstArrayView faces_infos,Int64ArrayView face_info2, Int64Array& node_uids);
  void _fillEdgeInfo2(Integer nb_edge,Int64ConstArrayView edges_infos,Int64ArrayView edge_info2, Int64Array& node_uids);

  void _fillEdgeNewInfoNew(Integer nb_edge,Int64ConstArrayView edges_infos,Int64ArrayView edge_new_infos);
  void _fillNodeNewInfo(Int64ConstArrayView node_uids,Int64ArrayView node_infos2);
  void _fillCellNewInfoNew(Integer nb_cell,Int64ConstArrayView cells_infos,Int64Array& cell_infos2, const std::map<Int64,Int64SharedArray>& cell_to_face_connectivity_info, const std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map);
  void _fillFaceNewInfoNew(Integer nb_face,Int64ConstArrayView faces_infos,Int64Array& face_infos2, const std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map);
  void _fillItemInfo2(ItemDataList& item_data_list, Int64ConstArrayView cells_infos);

  void _initNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _initEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _initFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _appendNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _appendEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _appendFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info);
  void _fillNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info, bool is_source_item_relation_data_emtpy);
  void _fillEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView edges_info, bool is_source_item_relation_data_emtpy);
  void _fillFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info, bool is_source_item_relation_data_emtpy);
  void _fillItemRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView source_item_types, bool is_source_item_relation_data_emtpy);
  void _initEmptyRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                              Integer approx_relation_size,
                              IItemFamily const* target_family);
  void _appendInitializedRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                                      Integer approx_relation_size,
                                      IItemFamily const* target_family);

  Int64 _findFaceUniqueId(Int64ConstArrayView work_face_sorted_nodes, NodeInFaceSet& face_nodes_set);
  Int64 _findFaceInFaceNodesSet(const NodeInFaceSet& face_nodes_set,Integer index,Int64ConstArrayView face_nodes, NodeInFacePtr node);
  void _addFaceNodes(NodeInFaceSet& face_nodes_set, Int64ConstArrayView face_nodes, Int64 face_uid);
  void _addFaceInFaceNodesSet(NodeInFaceSet& face_nodes_set,Integer index,Int64ConstArrayView face_nodes, NodeInFacePtr node, Int64 face_uid);
  NodeInFacePtr& _insertNode(NodeInFaceSet& face_nodes_set, Int64 inserted_node_uid);
  void _addItemsOrRelations(ItemDataList& info_list, IItemFamilyNetwork::eSchedulingOrder family_graph_traversal_order);

 private:
  
  DynamicMesh* m_mesh; //!< Maillage associé
  ItemTypeMng* m_item_type_mng = nullptr;
  
  Integer m_connectivity = 0; //!< Info de connectivité du maillage courant
  bool m_has_edge = false; //!< Info sur la présence d'arête (accèlere l'accès à la connectivité générale)
  
  //! AMR
  bool m_has_amr;

  bool m_verbose = false; //!< Vrai si affiche messages

  //! Outils de construction du maillage
  OneMeshItemAdder* m_one_mesh_item_adder = nullptr;   //!< Outil pour ajouter un élément au maillage
  GhostLayerBuilder* m_ghost_layer_builder = nullptr;  //!< Outil pour construire les éléments fantômes
  FaceUniqueIdBuilder* m_face_unique_id_builder = nullptr;
  EdgeUniqueIdBuilder* m_edge_unique_id_builder = nullptr;

  Int64 m_face_uid_pool = 0; //!< Numéro du uniqueId() utilisé pour générer les faces
  Int64 m_edge_uid_pool = 0; //!< Numéro du uniqueId() utilisé pour générer les edges
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
