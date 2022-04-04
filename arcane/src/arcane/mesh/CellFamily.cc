﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamily.cc                                               (C) 2000-2017 */
/*                                                                           */
/* Famille de mailles.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/CellFamily.h"

#include "arcane/mesh/CellMerger.h"

#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"
#include "arcane/ItemInternalEnumerator.h"
#include "arcane/Connectivity.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"
#include "arcane/mesh/NewWithLegacyConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellFamily::TopologyModifier
: public AbstractItemFamilyTopologyModifier
{
 public:
  TopologyModifier(CellFamily* f)
  :  AbstractItemFamilyTopologyModifier(f), m_true_family(f){}
 public:
  void replaceNode(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceNode(item_lid,index,new_lid);
  }
  void replaceEdge(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceEdge(item_lid,index,new_lid);
  }
  void replaceFace(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceFace(item_lid,index,new_lid);
  }
  void replaceHParent(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceHParent(item_lid,index,new_lid);
  }
  void replaceHChild(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceHChild(item_lid,index,new_lid);
  }
 private:
  CellFamily* m_true_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellFamily::
CellFamily(IMesh* mesh,const String& name)
: ItemFamily(mesh,IK_Cell,name)
, m_node_prealloc(0)
, m_edge_prealloc(0)
, m_face_prealloc(0)
, m_mesh_connectivity(0)
, m_node_family(nullptr)
, m_edge_family(nullptr)
, m_face_family(nullptr)
, m_node_connectivity(nullptr)
, m_edge_connectivity(nullptr)
, m_face_connectivity(nullptr)
, m_hparent_connectivity(nullptr)
{
  _setTopologyModifier(new TopologyModifier(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellFamily::
~CellFamily()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
build()
{
  ItemFamily::build();

  m_node_family = ARCANE_CHECK_POINTER(dynamic_cast<NodeFamily*>(m_mesh->nodeFamily()));
  m_edge_family = ARCANE_CHECK_POINTER(dynamic_cast<EdgeFamily*>(m_mesh->edgeFamily()));
  m_face_family = ARCANE_CHECK_POINTER(dynamic_cast<FaceFamily*>(m_mesh->faceFamily()));

  IItemFamilyNetwork* network = m_mesh->itemFamilyNetwork();
  if (m_mesh->useMeshItemFamilyDependencies()) { // temporary to fill legacy, even with family dependencies
    auto* nc = network->getConnectivity(this,m_node_family,connectivityName(this,m_node_family));
    using NodeNetwork = NewWithLegacyConnectivityType<CellFamily,NodeFamily>::type;
    m_node_connectivity = ARCANE_CHECK_POINTER(dynamic_cast<NodeNetwork*>(nc));
    using EdgeNetwork = NewWithLegacyConnectivityType<CellFamily,EdgeFamily>::type;
    auto* ec = network->getConnectivity(this,m_edge_family,connectivityName(this,m_edge_family));
    m_edge_connectivity = ARCANE_CHECK_POINTER(dynamic_cast<EdgeNetwork*>(ec));
    using FaceNetwork = NewWithLegacyConnectivityType<CellFamily,FaceFamily>::type;
    auto* fc = network->getConnectivity(this,m_face_family,connectivityName(this,m_face_family));
    m_face_connectivity = ARCANE_CHECK_POINTER(dynamic_cast<FaceNetwork*>(fc));
  }
  else{
    m_node_connectivity = new NodeConnectivity(this,m_node_family,"CellNode");
    m_edge_connectivity = new EdgeConnectivity(this,m_edge_family,"CellEdge");
    m_face_connectivity = new FaceConnectivity(this,m_face_family,"CellFace");
  }
  m_hparent_connectivity = new HParentConnectivity(this,this,"HParentCell");
  m_hchild_connectivity = new HChildConnectivity(this,this,"HChildCell");

  _addConnectivitySelector(m_node_connectivity);
  _addConnectivitySelector(m_edge_connectivity);
  _addConnectivitySelector(m_face_connectivity);
  _addConnectivitySelector(m_hparent_connectivity);
  _addConnectivitySelector(m_hchild_connectivity);

  _buildConnectivitySelectors();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void CellFamily::
_createOne(ItemInternal* item,Int64 uid,ItemTypeInfo* type)
{
  ItemLocalId item_lid(item);
  m_item_internal_list->cells = _itemsInternal();
  _allocateInfos(item,uid,type);
  auto nc = m_node_connectivity->trueCustomConnectivity();
  if (nc)
    nc->addConnectedItems(item_lid,type->nbLocalNode());
  if (m_edge_prealloc!=0){
    auto ec = m_edge_connectivity->trueCustomConnectivity();
    if (ec)
      ec->addConnectedItems(item_lid,type->nbLocalEdge());
  }
  auto fc = m_face_connectivity->trueCustomConnectivity();
  if (fc)
    fc->addConnectedItems(item_lid,type->nbLocalFace());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* CellFamily::
allocOne(Int64 uid,ItemTypeInfo* type, MeshInfos& mesh_info)
{
  ++mesh_info.nbCell();
  return allocOne(uid,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* CellFamily::
findOrAllocOne(Int64 uid,ItemTypeInfo* type,MeshInfos& mesh_info, bool& is_alloc)
{
  auto cell = findOrAllocOne(uid,type,is_alloc);
  if (is_alloc)
    ++mesh_info.nbCell();
  return cell;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* CellFamily::
allocOne(Int64 uid,ItemTypeInfo* type)
{
  ItemInternal* item = _allocOne(uid);
  _createOne(item,uid,type);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* CellFamily::
findOrAllocOne(Int64 uid,ItemTypeInfo* type,bool& is_alloc)
{
  // ARCANE_ASSERT((type->typeId() != IT_Line2),("Bad new 1D cell uid=%ld", uid)); // Assertion OK, but expensive ?
  ItemInternal* item = _findOrAllocOne(uid,is_alloc);
  if (is_alloc)
    _createOne(item,uid,type);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
preAllocate(Integer nb_item)
{
  Integer base_mem = ItemSharedInfo::COMMON_BASE_MEMORY;
  Integer mem = base_mem * (nb_item+1);
  info() << "Cellfamily: reserve=" << mem;
  _reserveInfosMemory(mem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
computeSynchronizeInfos()
{
  debug() << "Creating the list of ghosts cells";
  ItemFamily::computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_removeSubItems(Cell cell)
{
  ItemLocalId cell_lid(cell.localId());

  // Il faut d'abord supprimer les faces, les arêtes puis ensuite les noeuds
  // voir les remarques sur _removeOne dans les familles.
  // NOTE GG: ce n'est normalement plus obligatoire de le faire dans un ordre
  // fixe car ces méthodes ne suppriment pas les entités. La destruction
  // est faire lors de l'appel à removeNotConnectedSubItems().
  for( Face face : cell.faces() )
    m_face_family->removeCellFromFace(face.internal(),cell_lid);
  for( Edge edge : cell.edges() )
    m_edge_family->removeCellFromEdge(edge.internal(),cell_lid);
  for( Node node : cell.nodes() )
    m_node_family->removeCellFromNode(node.internal(),cell_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
removeCell(ItemInternal* icell)
{
#ifdef ARCANE_CHECK
  _checkValidItem(icell);
  if (icell->isSuppressed())
    ARCANE_FATAL("Cell '{0}' is already removed",icell->uniqueId());
#endif
  // TODO: supprimer les faces et arêtes connectées.
  _removeSubItems(icell);
  _removeNotConnectedSubItems(icell);
  //! AMR
   if (icell->level() >0){
     _removeParentCellToCell(icell);
     Cell cell(icell);
     Cell parent_cell= cell.hParent();
     _removeChildCellToCell(parent_cell.internal(),icell);
   }
  _removeOne(icell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
detachCell(ItemInternal* icell)
{
#ifdef ARCANE_CHECK
  _checkValidItem(icell);
  if (icell->isSuppressed())
    ARCANE_FATAL("Cell '{0}' is already removed",icell->uniqueId());
#endif /* ARCANE_CHECK */

  _removeSubItems(icell);
  //! AMR
  if (icell->level() >0){
    _removeParentCellToCell(icell);
    Cell cell(icell);
    Cell parent_cell= cell.hParent();
    _removeChildCellToCell(parent_cell.internal(),icell);
  }
  _detachOne(icell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
detachCells2(Int32ConstArrayView cells_local_id)
{
  // Implemented in ItemFamily. Even if only cells are detached, the implementation
  // is not CellFamily specific, thx to ItemFamilyNetwork
  _detachCells2(cells_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Supprime les sous-entités de la maille qui ne sont connectées
 * à aucune maille.
 */
void CellFamily::
_removeNotConnectedSubItems(Cell cell)
{
  // L'ordre (faces, puis arêtes puis noeuds) est important.
  // Ne pas changer.

  // Supprime les faces de la maille qui ne sont plus connectées
  for( Face face : cell.faces() )
    m_face_family->removeFaceIfNotConnected(face.internal());

  // Supprime les arêtes de la maille qui ne sont plus connectées
  for( Edge edge : cell.edges() )
    m_edge_family->removeEdgeIfNotConnected(edge.internal());

  // on supprime les noeuds de la maille qui ne sont plus connectés
  for( Node node : cell.nodes() )
    m_node_family->removeNodeIfNotConnected(node.internal());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
removeDetachedCell(ItemInternal* cell)
{
  _removeNotConnectedSubItems(cell);

  // on supprime la maille
  _removeDetachedOne(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost)
{
  ARCANE_UNUSED(keep_ghost);
  for( Integer i=0, is=local_ids.size(); i<is; ++i ){
    removeCell(m_item_internal_list->cells[local_ids[i]]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
mergeItems(Int32 local_id1,Int32 local_id2)
{
  ItemInternal* icell1 = m_item_internal_list->cells[local_id1];
  ItemInternal* icell2 = m_item_internal_list->cells[local_id2];

  CellMerger cm(traceMng());
  cm.merge(icell1, icell2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CellFamily::
getMergedItemLID(Int32 local_id1,Int32 local_id2)
{
  ItemInternal* icell1 = m_item_internal_list->cells[local_id1];
  ItemInternal* icell2 = m_item_internal_list->cells[local_id2];

  CellMerger cm(traceMng());
  ItemInternal* remaining_cell = cm.getItemInternal(icell1, icell2);

  return (remaining_cell==icell1) ? local_id1 : local_id2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace le noeud d'index \a index de la maille \a cell avec
 * celui de localId() \a node.
 */
void CellFamily::
replaceNode(ItemLocalId cell,Integer index,ItemLocalId node)
{
  m_node_connectivity->replaceItem(cell,index,node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace l'arête d'index \a index de la maille \a cell avec
 * celle de localId() \a edge.
 */
void CellFamily::
replaceEdge(ItemLocalId cell,Integer index,ItemLocalId edge)
{
  m_edge_connectivity->replaceItem(cell,index,edge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace la face d'index \a index de la maille \a cell avec
 * celle de localId() \a face.
 */
void CellFamily::
replaceFace(ItemLocalId cell,Integer index,ItemLocalId face)
{
  m_face_connectivity->replaceItem(cell,index,face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
replaceHChild(ItemLocalId cell,Integer index,ItemLocalId child_cell)
{
  m_hchild_connectivity->replaceItem(cell,index,child_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
replaceHParent(ItemLocalId cell,Integer index,ItemLocalId parent_cell)
{
  m_hparent_connectivity->replaceItem(cell,index,parent_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
setConnectivity(const Integer c)
{
  m_mesh_connectivity = c;
  m_node_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Cell,IK_Node);
  m_node_connectivity->setPreAllocatedSize(m_node_prealloc);
  // Les arêtes n'existent que en dimension 3.
  if (mesh()->dimension()==3){
    Integer edge_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Cell,IK_Edge);
    m_edge_connectivity->setPreAllocatedSize(edge_prealloc);
    if (Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasEdge))
      m_edge_prealloc = edge_prealloc;
  }
  m_face_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Cell,IK_Face);
  m_face_connectivity->setPreAllocatedSize(m_face_prealloc);
  debug() << "Family " << name() << " prealloc "
          << m_node_prealloc << " by node, "
          << m_edge_prealloc << " by edge, "
          << m_face_prealloc << " by face.";
}
//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_addParentCellToCell(ItemInternal* cell,ItemInternal* parent_cell)
{
  m_hparent_connectivity->addConnectedItem(ItemLocalId(cell),ItemLocalId(parent_cell));
  //_updateSharedInfoAdded(cell,0,0,0,1,0);
  //cell->setHParent(0,parent_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_addChildCellToCell(ItemInternal* iparent_cell,Integer rank,ItemInternal* child_cell)
{
  Cell parent_cell(iparent_cell);
  // NOTE GG: Cette méthode ne semble fonctionner que si \a rank
  // correspond parent_cell->nbHChildren().
  // Et dans ce cas il n'est pas nécessaire de faire 2 appels.
  m_hchild_connectivity->addConnectedItem(parent_cell,ItemLocalId(NULL_ITEM_LOCAL_ID));
  auto x = _topologyModifier();
  x->replaceHChild(ItemLocalId(iparent_cell),rank,ItemLocalId(child_cell));
  iparent_cell->addFlags(ItemInternal::II_Inactive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_addChildrenCellsToCell(ItemInternal* parent_cell,Int32ConstArrayView children_cells_lid)
{
  Integer nb_children = children_cells_lid.size();
  auto c = m_hchild_connectivity->trueCustomConnectivity();
  if (c){
    ItemLocalId item_lid(parent_cell);
    c->addConnectedItems(item_lid,nb_children);
  }
  _updateSharedInfoAdded(parent_cell);

  auto x = _topologyModifier();
  for( Integer i=0; i<nb_children; ++i )
    x->replaceHChild(ItemLocalId(parent_cell),i,ItemLocalId(children_cells_lid[i]));

  parent_cell->addFlags(ItemInternal::II_Inactive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_removeParentCellToCell(ItemInternal* cell)
{
  m_hparent_connectivity->removeConnectedItems(ItemLocalId(cell));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_removeChildCellToCell(ItemInternal* parent_cell,ItemInternal* cell)
{
  //_updateSharedInfoRemoved(parent_cell,0,0,0,0,1);
  m_hchild_connectivity->removeConnectedItem(ItemLocalId(parent_cell),ItemLocalId(cell));
  parent_cell->removeFlags(ItemInternal::II_Inactive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamily::
_removeChildrenCellsToCell(ItemInternal* parent_cell)
{
  m_hchild_connectivity->removeConnectedItems(ItemLocalId(parent_cell));
  //Integer nb_children = parent_cell->nbHChildren();
  //_updateSharedInfoRemoved(parent_cell,0,0,0,0,nb_children);
  parent_cell->removeFlags(ItemInternal::II_Inactive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
