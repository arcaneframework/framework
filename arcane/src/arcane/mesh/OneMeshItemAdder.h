// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*                                                                           */
/* Outil de création d'une maille                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ONEITEMADDER_H
#define ARCANE_MESH_ONEITEMADDER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/FullItemInfo.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/MeshInfos.h"

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class DynamicMeshIncrementalBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneMeshItemAdder
  : public TraceAccessor
{
private:
  
  // Classe servant à rendre compatible les données FullCellInfo
  // et les données fragmentées de description d'une maille
  class CellInfoProxy;
  
public:
  
  OneMeshItemAdder(DynamicMeshIncrementalBuilder* mesh_builder);
  ~OneMeshItemAdder() {}
  
  ItemInternal* addOneNode(Int64 node_uid,Int32 owner);

  // DEPRECATED
  ItemInternal* addOneFace(Int64 a_face_uid, 
                           Int64ConstArrayView a_node_list, 
                           Integer a_type);
 
  ItemInternal* addOneFace(ItemTypeInfo* type,
                           Int64 face_uid,
                           Int32 sub_domain_id,
                           Int64ConstArrayView nodes_uid);
  
  ItemInternal* addOneEdge(Int64 edge_uid,
                           Integer sub_domain_id,
                           Int64ConstArrayView nodes_uid);
  
  ItemInternal* addOneCell(ItemTypeInfo* type,
                           Int64 cell_uid,
                           Integer sub_domain_id,
                           Int64ConstArrayView nodes_uid,
                           bool allow_build_face);
  
  ItemInternal* addOneParentItem(const Item & item, 
                                 const eItemKind submesh_kind, 
                                 const bool fatal_on_existing_item = true);
 
  ItemInternal* addOneCell(const FullCellInfo& cell_info);

  ItemInternal* addOneItem(IItemFamily* family,
                           IItemFamilyModifier* family_modifier,
                           ItemTypeInfo* type,
                           Int64 item_uid,
                           Integer item_owner,
                           Integer sub_domain_id,
                           Integer nb_connected_family,
                           Int64ConstArrayView connectivity_info);

  ItemInternal* addOneItem2(IItemFamily* family,
                            IItemFamilyModifier* family_modifier,
                            ItemTypeInfo* type,
                            Int64 item_uid,
                            Integer item_owner,
                            Integer sub_domain_id,
                            Integer nb_connected_family,
                            Int64ConstArrayView connectivity_info);
   
  Integer nbNode() const { return m_mesh_info.getNbNode(); }
  Integer nbFace() const { return m_mesh_info.getNbFace(); }
  Integer nbCell() const { return m_mesh_info.getNbCell(); }
  Integer nbEdge() const { return m_mesh_info.getNbEdge(); }

  void setNextFaceUid(Int64 face_uid) { m_next_face_uid =  face_uid; }
  void setNextEdgeUid(Int64 edge_uid) { m_next_edge_uid =  edge_uid; }

  Int64 nextFaceUid() const { return m_next_face_uid; }
  Int64 nextEdgeUid() const { return m_next_edge_uid; }

private:
  
  template<typename CellInfo>
  ItemInternal* _addOneCell(const CellInfo& cell_info);

  template<typename CellInfo>
  void _addNodesToCell(ItemInternal* cell,
                       const CellInfo& cell_info);
  
  template<typename CellInfo>
  bool _isReorder(Integer i_face, 
                  const ItemTypeInfo::LocalFace& lf, 
                  const CellInfo& cell_info);

  template<typename CellInfo>
  ItemInternal* _findInternalFace(Integer i_face, 
                                  const CellInfo& cell_info, 
                                  bool& is_add);

  template<typename CellInfo>
  ItemInternal* _findInternalEdge(Integer i_edge, 
                                  const CellInfo& cell_info, 
                                  Int64 first_node, 
                                  Int64 second_node, 
                                  bool& is_add);
  template<typename CellInfo>
  void _AMR_Patch(ItemInternal* cell, const CellInfo& cell_info);

  void _clearConnectivity(ItemLocalId item, IIncrementalItemConnectivity* connectivity);
  void _clearReverseConnectivity(ItemLocalId item, IIncrementalItemConnectivity* connectivity, IIncrementalItemConnectivity* reverse_connectivity);
  void _printRelations(ItemInternal* item);

private:
 
  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;
 
  CellFamily& m_cell_family;
  NodeFamily& m_node_family;
  FaceFamily& m_face_family;
  EdgeFamily& m_edge_family;
  
  ItemTypeMng* m_item_type_mng;
 
  MeshInfos m_mesh_info;//!<  Info générale sur le maillage (numéro de sous-domaine, nombre d'items...)
  
  Int64 m_next_face_uid; //!< Numéro du uniqueId() suivant utilisé pour générer les faces
  Int64 m_next_edge_uid; //!< Numéro du uniqueId() suivant utilisé pour générer les arêtes
  
  //! Tableaux de travail
  Int64UniqueArray m_work_face_sorted_nodes;
  Int64UniqueArray m_work_face_orig_nodes_uid;
  Int64UniqueArray m_work_edge_sorted_nodes;
  Int64UniqueArray m_work_edge_orig_nodes_uid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_ONEITEMADDER_H */
