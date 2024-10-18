// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelAMRConsistency.h                                    (C) 2000-2024 */
/*                                                                           */
/* Gestion de la consistance de l'AMR en parallèle.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_PARALLELAMRCONSISTENCY_H
#define ARCANE_MESH_PARALLELAMRCONSISTENCY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/Real3.h"

#include "arcane/IMesh.h"
#include "arcane/ItemGroup.h"
#include "arcane/Item.h"
#include "arcane/VariableTypes.h"

#include "arcane/mesh/DynamicMeshKindInfos.h"

#include "arcane/utils/PerfCounterMng.h"

#include <unordered_set>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR

class FaceInfoMng
{
public:
  Integer size() const
  {
    return m_nodes_unique_id.size();
  }
  void add(ItemUniqueId node_unique_id)
  {
    m_nodes_unique_id.add(node_unique_id);
  }
  void set(Integer i, ItemUniqueId node_unique_id)
  {
    m_nodes_unique_id[i] = node_unique_id;
  }
public:
  SharedArray<ItemUniqueId> m_nodes_unique_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeInfo
{
 public:
  NodeInfo() : m_unique_id(NULL_ITEM_ID), m_owner(A_NULL_RANK)
  {
  }
  NodeInfo(ItemUniqueId node_uid, Integer aowner) :
    m_unique_id(node_uid), m_owner(aowner)
  {
  }
 public:
  ItemUniqueId uniqueId() const
  {
    return m_unique_id;
  }
  void addConnectedFace(ItemUniqueId uid)
  {
    for (Integer i = 0, is = m_connected_active_faces.size(); i < is; ++i)
      if (m_connected_active_faces[i] == uid)
        return;
    m_connected_active_faces.add(uid);
  }
  void setCoord(Real3 coord)
  {
    m_coord = coord;
  }
  Real3 getCoord() const
  {
    return m_coord;
  }
  Integer owner() const
  {
    return m_owner;
  }
private:
  //! Numéro de ce noeud
  ItemUniqueId m_unique_id;
  //! propriétaire du noeud
  Integer m_owner;
  //! Coordonnées de ce noeud
  Real3 m_coord;

public:

  //! Liste des uniqueId() des faces actives auquel ce noeud peut être connecté
  SharedArray<ItemUniqueId> m_connected_active_faces;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos sur une Face active.
 *
 * Cet objet peut être copié mais ne doit pas être conservé une
 * fois le gestionnnaire \a m_mng associé détruit.
 */
class FaceInfo
{
public:
  FaceInfo() :
    m_unique_id(NULL_ITEM_ID), m_owner(A_NULL_RANK), m_nb_node(0), m_data_index(-1), m_mng(0)
  {
  }
  FaceInfo(
      ItemUniqueId unique_id,
      ItemUniqueId cell_unique_id,
      Integer nb_node,
      Integer owner,
      Integer data_index,
      FaceInfoMng* mng) :
    m_unique_id(unique_id), m_cell_unique_id(cell_unique_id), m_owner(owner), m_nb_node(nb_node),
        m_data_index(data_index), m_mng(mng)
  {
  }
public:
  ItemUniqueId uniqueId() const
  {
    return m_unique_id;
  }
  ItemUniqueId cellUniqueId() const
  {
    return m_cell_unique_id;
  }
  Integer nbNode() const
  {
    return m_nb_node;
  }
  ItemUniqueId nodeUniqueId(Integer i) const
  {
    return m_mng->m_nodes_unique_id[m_data_index + i];
  }
  void setNodeUniqueId(Integer i, const ItemUniqueId& uid)
  {
    m_mng->m_nodes_unique_id[m_data_index + i] = uid;
  }
  Integer owner() const
  {
    return m_owner;
  }
  void setCenter(Real3 center)
  {
    m_center = center;
  }
  Real3 center() const
  {
    return m_center;
  }
  Integer getDataIndex()
  {
    return m_data_index;
  }
private:
  ItemUniqueId m_unique_id;
  ItemUniqueId m_cell_unique_id;
  Integer m_owner;
  Integer m_nb_node;
  Real3 m_center;
public:
  Integer m_data_index;
  FaceInfoMng* m_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos sur une Face active.
 *
 * Cet objet peut être copié mais ne doit pas être conservé une
 * fois le gestionnnaire \a m_mng associé détruit.
 */
class FaceInfo2
{
 public:
  FaceInfo2()
  : m_unique_id(NULL_ITEM_ID), m_owner(A_NULL_RANK) { }
  FaceInfo2(ItemUniqueId unique_id, Integer aowner)
  : m_unique_id(unique_id), m_owner(aowner) { }
 public:
  ItemUniqueId uniqueId() const
  {
    return m_unique_id;
  }
  Integer owner() const
  {
    return m_owner;
  }
  void setCenter(Real3 center)
  {
    m_center = center;
  }
  Real3 center() const
  {
    return m_center;
  }

 private:
  ItemUniqueId m_unique_id;
  Int32 m_owner;
  Real3 m_center;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelAMRConsistency
: public TraceAccessor
{
 public:

  typedef HashTableMapT<ItemUniqueId, NodeInfo> NodeInfoList;
  typedef HashTableMapT<ItemUniqueId, FaceInfo> FaceInfoMap;
  typedef HashTableMapT<ItemUniqueId, FaceInfo2> FaceInfoMap2;
  typedef HashTableMapEnumeratorT<ItemUniqueId, NodeInfo> NodeInfoListEnumerator;
  typedef HashTableMapEnumeratorT<ItemUniqueId, FaceInfo2> FaceInfo2MapEnumerator;


  typedef std::unordered_set<Int64> ItemUidSet;
  typedef std::unordered_map<Int64,Item> ItemMap;
  typedef std::pair<Int64,Item> ItemMapValue;


#ifdef ACTIVATE_PERF_COUNTER
  struct PerfCounter
  {
      typedef enum {
        INIT,
        COMPUTE,
        GATHERFACE,
        UPDATE,
        REHASH,
        ENDUPDATE,
        NbCounters
      }  eType ;

      static const std::string m_names[NbCounters] ;
  } ;
#endif
public:
  ParallelAMRConsistency(IMesh* mesh);

public:
  void init() ;
  void invalidate() ;
  bool isUpdated() const {
    return m_is_updated ;
  }
  void update() {
    if(!m_is_updated) init() ;
  }
  void makeNewItemsConsistent(NodeMapCoordToUid& node_finder, FaceMapCoordToUid& face_finder);
  void makeNewItemsConsistent2(MapCoordToUid& node_finder, MapCoordToUid& face_finder);
  void changeOwners(Int64UniqueArray linked_cells, Int32UniqueArray linked_owers);
  void changeOwnersOld();

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter>& getPerfCounter() {
    return m_perf_counter ;
  }
#endif
private:

  IMesh* m_mesh;
  VariableNodeReal3 m_nodes_coord;
  FaceInfoMng m_face_info_mng;
  NodeInfoList m_nodes_info;
  NodeInfoList m_active_nodes;
  FaceInfoMap m_active_faces;
  FaceInfoMap2 m_active_faces2;
  String m_active_face_name;
  FaceGroup m_active_face_group;

  bool m_is_updated ;
  UniqueArray<Int64> m_shared_face_uids ;
  UniqueArray<Int64> m_connected_shared_face_uids ;

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter> m_perf_counter ;
#endif
private:

  bool _isInsideFace(const FaceInfo& face, Real3 point);

  void _gatherFaces(ConstArrayView<ItemUniqueId> faces_to_send,
                    ConstArrayView<ItemUniqueId> nodes_to_send,
                    FaceInfoMap& face_map,
                    MapCoordToUid& node_finder,
                    MapCoordToUid& face_finder,
                    ItemUidSet& updated_face_uids,
                    ItemUidSet& updated_node_uids);

  void _update(Array<ItemUniqueId>& nodes_unique_id, NodeInfoList const& nodes_info) ;

  void _gatherItems(ConstArrayView<ItemUniqueId> nodes_to_send,
                    ConstArrayView<ItemUniqueId> faces_to_send,
                    NodeInfoList& node_map,
                    FaceInfoMap2& face_map,
                    MapCoordToUid& node_finder,
                    MapCoordToUid& face_finder);

  void _gatherAllNodesInfo();

  void _printFaces(std::ostream& o, FaceInfoMap& face_map);

  void _addFaceToList(Face face, FaceInfoMap& face_map);

  void _addFaceToList2(Face face, FaceInfoMap2& face_map);
  void _addNodeToList(Node node, NodeInfoList& node_map);
  bool _hasSharedNodes(Face face);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* PARALLELAMRCONSISTENCY_H_ */
