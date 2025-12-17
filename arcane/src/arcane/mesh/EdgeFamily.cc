// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeFamily.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Famille d'arêtes.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/EdgeFamily.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/NodesOfItemReorderer.h"

#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"
#include "arcane/mesh/NewWithLegacyConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EdgeFamily::TopologyModifier
: public AbstractItemFamilyTopologyModifier
{
 public:
  TopologyModifier(EdgeFamily* f)
  :  AbstractItemFamilyTopologyModifier(f), m_true_family(f){}
 public:
  void replaceNode(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceNode(item_lid,index,new_lid);
  }
 private:
  EdgeFamily* m_true_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeFamily::
EdgeFamily(IMesh* mesh,const String& name)
: ItemFamily(mesh,IK_Edge,name)
{
  _setTopologyModifier(new TopologyModifier(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeFamily::
~EdgeFamily()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
build()
{
  ItemFamily::build();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  m_edge_type = itm->typeFromId(IT_Line2);

  m_node_family = ARCANE_CHECK_POINTER(dynamic_cast<NodeFamily*>(m_mesh->nodeFamily()));

  if (m_mesh->useMeshItemFamilyDependencies()) // temporary to fill legacy, even with family dependencies
  {
    m_node_connectivity = dynamic_cast<NewWithLegacyConnectivityType<EdgeFamily,NodeFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->nodeFamily(),connectivityName(this,mesh()->nodeFamily())));
    m_face_connectivity = dynamic_cast<NewWithLegacyConnectivityType<EdgeFamily,FaceFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->faceFamily(),connectivityName(this,mesh()->faceFamily())));
    m_cell_connectivity = dynamic_cast<NewWithLegacyConnectivityType<EdgeFamily,CellFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->cellFamily(),connectivityName(this,mesh()->cellFamily())));
  }
  else
  {
    m_node_connectivity = new NodeConnectivity(this,mesh()->nodeFamily(),"EdgeNode");
    m_face_connectivity = new FaceConnectivity(this,mesh()->faceFamily(),"EdgeFace");
    m_cell_connectivity = new CellConnectivity(this,mesh()->cellFamily(),"EdgeCell");
  }

  _addConnectivitySelector(m_node_connectivity);
  _addConnectivitySelector(m_face_connectivity);
  _addConnectivitySelector(m_cell_connectivity);

  _buildConnectivitySelectors();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void EdgeFamily::
_createOne(ItemInternal* item,Int64 uid)
{
  m_item_internal_list->edges = _itemsInternal();
  _allocateInfos(item,uid,m_edge_type);
  auto nc = m_node_connectivity->trueCustomConnectivity();
  if (nc)
    nc->addConnectedItems(ItemLocalId(item),2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Alloue une arête de numéro unique \a uid. Ajout générique d'item.
* Cette version est faite pour être appelée dans un bloc générique ignorant le type
 * de l'item. La mise à jour du nombre d'item du maillage est donc fait dans cette méthode,
 * et non dans le bloc appelant.
 */
Item EdgeFamily::
allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info)
{
  ARCANE_ASSERT((type_id == IT_Line2),(""));
  ARCANE_UNUSED(type_id);
  ++mesh_info.nbEdge();
  return allocOne(uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Alloue une arête de numéro unique \a uid.
 */
ItemInternal* EdgeFamily::
allocOne(Int64 uid)
{
  ItemInternal* item = _allocOne(uid);
  _createOne(item,uid);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère ou alloue une arête de numéro unique \a uid et de type \a type.Ajout générique d'item.
 *
 * Cette version est faite pour être appelée dans un bloc générique ignorant le type
 * de l'item. La mise à jour du nombre d'item du maillage est donc fait dans cette méthode,
 * et non dans le bloc appelant.
 *
 * Si une arête de numéro unique \a uid existe déjà, la retourne. Sinon,
 * l'arête est créée. \a is_alloc est vrai si l'arête vient d'être créée.
 */
Item EdgeFamily::
findOrAllocOne(Int64 uid,ItemTypeId type_id,MeshInfos& mesh_info, bool& is_alloc)
{
  ARCANE_ASSERT((type_id == IT_Line2),(""));
  ARCANE_UNUSED(type_id);
  auto edge = findOrAllocOne(uid,is_alloc);
  if (is_alloc)
    ++mesh_info.nbEdge();
  return edge;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère ou alloue une arête de numéro unique \a uid et de type \a type.
 *
 * Si une arête de numéro unique \a uid existe déjà, la retourne. Sinon,
 * l'arête est créée. \a is_alloc est vrai si l'arête vient d'être créée.
 */
ItemInternal* EdgeFamily::
findOrAllocOne(Int64 uid,bool& is_alloc)
{
  ItemInternal* item = _findOrAllocOne(uid,is_alloc);
  if (is_alloc)
    _createOne(item,uid);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
preAllocate(Integer nb_item)
{
  if (m_has_edge) {
    this->_preAllocate(nb_item,true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
computeSynchronizeInfos()
{
  debug() << "Creating the list of ghosts edges";
  ItemFamily::computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace le noeud d'index \a index de l'arête \a edge avec
 * celui de localId() \a node_lid.
 */
void EdgeFamily::
replaceNode(ItemLocalId edge,Integer index,ItemLocalId node)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToNode))
    return;
  m_node_connectivity->replaceItem(edge,index,node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
addCellToEdge(Edge edge,Cell new_cell)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToCell))
    return;
  _checkValidSourceTargetItems(edge,new_cell);
  m_cell_connectivity->addConnectedItem(ItemLocalId(edge),ItemLocalId(new_cell));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
addFaceToEdge(Edge edge,Face new_face)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToFace))
    return;
  _checkValidSourceTargetItems(edge,new_face);
  m_face_connectivity->addConnectedItem(ItemLocalId(edge),ItemLocalId(new_face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void EdgeFamily::
_removeEdge(Edge edge)
{
  for( Node node : edge.nodes() )
    m_node_family->removeEdgeFromNode(node,edge);
  _removeOne(edge);
  // On ne supprime pas ici les autres relations (face->edge,cell->edge)
  // Car l'ordre de suppression doit toujours être cell, face, edge, node
  // donc node est en dernier et tout est déjà fait
  // Par ailleurs, cela évite des problèmes de récursivité
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeCellFromEdge(Edge edge,ItemLocalId cell_to_remove_lid)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToCell))
    return;
  _checkValidItem(edge);
  m_cell_connectivity->removeConnectedItem(ItemLocalId(edge),cell_to_remove_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeFaceFromEdge(ItemLocalId edge,ItemLocalId face_to_remove)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToFace))
    return;
  m_face_connectivity->removeConnectedItem(edge,face_to_remove);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeEdgeIfNotConnected(Edge edge)
{
	_checkValidItem(edge);
	if (!edge.itemBase().isSuppressed() && edge.nbCell()==0){
    _removeEdge(edge);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
setConnectivity(const Integer c) 
{
  m_mesh_connectivity = c;
  m_has_edge = Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasEdge);
  if (m_has_edge) {
    m_node_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Edge,IK_Node);
    m_face_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Edge,IK_Face);
    m_cell_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Edge,IK_Cell);
  }
  debug() << "Family " << name() << " prealloc " 
          << m_node_prealloc << " by node, " 
          << m_face_prealloc << " by face, "
          << m_cell_prealloc << " by cell.";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
reorientEdgesIfNeeded()
{
  info() << "Reorient Edges family=" << fullName();
  // Réoriente les arêtes si nécessaire. Cela est le cas par exemple si on
  // a changé la numérotation des uniqueId() des noeuds.
  NodesOfItemReorderer reorderer(mesh()->itemTypeMng());
  SmallArray<Int32> new_nodes_lid;
  IncrementalItemConnectivity* true_connectivity = m_node_connectivity->trueCustomConnectivity();
  ENUMERATE_ (Edge, iedge, allItems()) {
    Edge edge = *iedge;
    ItemTypeId edge_type = edge.itemTypeId();
    Int32 nb_node = edge.nbNode();
    // Il faut que le premier noeud soit celui de plus petit uniqueId().
    // Si le type est ITI_Line3, le troisième noeud ne change pas.
    if (edge_type == ITI_Line2 || edge_type == ITI_Line3) {
      new_nodes_lid.resize(nb_node);
      for (Int32 i = 0; i < nb_node; ++i)
        new_nodes_lid[i] = edge.nodeId(i);
      Node node0 = edge.node(0);
      Node node1 = edge.node(1);
      if (node0.uniqueId() > node1.uniqueId()) {
        std::swap(new_nodes_lid[0], new_nodes_lid[1]);
      }
      true_connectivity->replaceConnectedItems(edge, new_nodes_lid);
    }
    else
      ARCANE_FATAL("Reorientation of edge of type '{0}' not yet supported", edge_type);
  }
  info() << "End Edge Reorientation";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
