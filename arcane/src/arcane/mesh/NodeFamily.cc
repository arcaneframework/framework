// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeFamily.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Famille de noeuds.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/NodeFamily.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/Properties.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"
#include "arcane/mesh/NewWithLegacyConnectivity.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/EdgeFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeFamily::TopologyModifier
: public AbstractItemFamilyTopologyModifier
{
 public:
  TopologyModifier(NodeFamily* f)
  :  AbstractItemFamilyTopologyModifier(f), m_true_family(f){}
 public:
  void replaceEdge(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceEdge(item_lid,index,new_lid);
  }
  void replaceFace(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceFace(item_lid,index,new_lid);
  }
  void replaceCell(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceCell(item_lid,index,new_lid);
  }
 private:
  NodeFamily* m_true_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeFamily::
NodeFamily(IMesh* mesh,const String& name)
: ItemFamily(mesh,IK_Node,name)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_SORT_FACE_AND_EDGE_OF_NODE", true)){
    m_is_sort_connected_faces_and_edges = v.value() !=0;
    info() << "Set sort faces and edges of nodes v?=" << m_is_sort_connected_faces_and_edges;
  }
  _setTopologyModifier(new TopologyModifier(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeFamily::
~NodeFamily()
{
  delete m_nodes_coords;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
build()
{
  ItemFamily::build();

  ItemTypeMng* itm = m_mesh->itemTypeMng();
  m_node_type = itm->typeFromId(IT_Vertex);
  if (m_parent_family)
    m_nodes_coords = nullptr;
  else
    m_nodes_coords = new VariableNodeReal3(VariableBuildInfo(mesh(),"NodeCoord"));

  m_face_family = ARCANE_CHECK_POINTER(dynamic_cast<FaceFamily*>(mesh()->faceFamily()));
  m_edge_family = ARCANE_CHECK_POINTER(dynamic_cast<EdgeFamily*>(mesh()->edgeFamily()));

  if (m_mesh->useMeshItemFamilyDependencies()) // temporary to fill legacy, even with family dependencies
  {
    m_edge_connectivity = dynamic_cast<NewWithLegacyConnectivityType<NodeFamily,EdgeFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->edgeFamily(),connectivityName(this,mesh()->edgeFamily())));
    m_face_connectivity = dynamic_cast<NewWithLegacyConnectivityType<NodeFamily,FaceFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,m_face_family,connectivityName(this,mesh()->faceFamily())));
    m_cell_connectivity = dynamic_cast<NewWithLegacyConnectivityType<NodeFamily,CellFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->cellFamily(),connectivityName(this,mesh()->cellFamily())));
  }
  else
  {
    m_edge_connectivity = new EdgeConnectivity(this,mesh()->edgeFamily(),"NodeEdge");
    m_face_connectivity = new FaceConnectivity(this,m_face_family,"NodeFace");
    m_cell_connectivity = new CellConnectivity(this,mesh()->cellFamily(),"NodeCell");
  }
  m_hparent_connectivity = new HParentConnectivity(this, this, "HParentNode");
  m_hchild_connectivity = new HChildConnectivity(this, this, "HChildNode");

  _addConnectivitySelector(m_edge_connectivity);
  _addConnectivitySelector(m_face_connectivity);
  _addConnectivitySelector(m_cell_connectivity);
  _addConnectivitySelector(m_hparent_connectivity);
  _addConnectivitySelector(m_hchild_connectivity);

  _buildConnectivitySelectors();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
preAllocate(Integer nb_item)
{
  this->_preAllocate(nb_item,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
_endAllocate()
{
  if (m_nodes_coords)
    m_nodes_coords->setUsed(true);
  ItemFamily::_endAllocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
addCellToNode(Node node,Cell new_cell)
{
  _checkValidSourceTargetItems(node,new_cell);
  Int32 cell_lid = new_cell.localId();
  m_cell_connectivity->addConnectedItem(ItemLocalId(node),ItemLocalId(cell_lid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
addFaceToNode(Node node,Face new_face)
{
  if (m_no_face_connectivity)
    return;

  _checkValidSourceTargetItems(node,new_face);
  m_face_connectivity->addConnectedItem(ItemLocalId(node),ItemLocalId(new_face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
addEdgeToNode(Node node,Edge new_edge)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_NodeToEdge))
    return;

  _checkValidSourceTargetItems(node,new_edge);
  m_edge_connectivity->addConnectedItem(ItemLocalId(node),ItemLocalId(new_edge));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
removeEdgeFromNode(ItemLocalId node,ItemLocalId edge_to_remove)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_NodeToEdge))
    return;
  m_edge_connectivity->removeConnectedItem(node,edge_to_remove);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
removeFaceFromNode(ItemLocalId node,ItemLocalId face_to_remove)
{
  if (m_no_face_connectivity)
    return;

  m_face_connectivity->removeConnectedItem(node,face_to_remove);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
removeCellFromNode(Node node,ItemLocalId cell_to_remove_lid)
{
	_checkValidItem(node);
  m_cell_connectivity->removeConnectedItem(ItemLocalId(node),cell_to_remove_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
removeNodeIfNotConnected(Node node)
{
	_checkValidItem(node);
	if (!node.itemBase().isSuppressed()){
		Integer nb_cell = node.nbCell();
		if (nb_cell == 0)
			_removeNode(node);
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
computeSynchronizeInfos()
{
  debug() << "Creating the ghosts nodes list";
  ItemFamily::computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void NodeFamily::
_removeNode(Node node)
{
  _removeOne(node);
  // On ne supprime pas ici les autres relations (edge->node, face->node)
  // Car l'ordre de suppression doit toujours être cell, face, edge, node
  // donc node est en dernier et tout est déjà fait
  // Par ailleurs, cela évite des problèmes de récursivité
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace la maille d'index \a index du noeud \a node avec
 * celle de localId() \a node_lid.
 */
void NodeFamily::
replaceCell(ItemLocalId node,Integer index,ItemLocalId cell)
{
  m_cell_connectivity->replaceItem(node,index,cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace l'arête d'index \a index du noeud \a node avec
 * celle de localId() \a face_lid.
 */
void NodeFamily::
replaceEdge(ItemLocalId node,Integer index,ItemLocalId edge)
{
  m_edge_connectivity->replaceItem(node,index,edge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace la face d'index \a index du noeud \a node avec
 * celle de localId() \a face_lid.
 */
void NodeFamily::
replaceFace(ItemLocalId node,Integer index,ItemLocalId face)
{
  m_face_connectivity->replaceItem(node,index,face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
setConnectivity(const Integer c) 
{
  m_mesh_connectivity = c;
  if (Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasEdge))
    m_edge_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Node,IK_Edge);
  m_face_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Node,IK_Face);
  m_cell_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Node,IK_Cell);
  m_face_connectivity->setPreAllocatedSize(m_face_prealloc);
  m_cell_connectivity->setPreAllocatedSize(m_cell_prealloc);
  debug() << "Family " << name() << " prealloc " 
          << m_edge_prealloc << " by edge, " 
          << m_face_prealloc << " by face, "
          << m_cell_prealloc << " by cell.";
  m_no_face_connectivity = !Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_NodeToFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeFamily::ItemCompare2
{
 public:

  explicit ItemCompare2(const ItemInfoListView& items)
  : m_items(items)
  {
  }

 public:

  bool operator()(Int32 item1,Int32 item2) const
  {
    return m_items.uniqueId(item1) < m_items.uniqueId(item2);
  }

 private:

  ItemInfoListView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeFamily::ItemCompare3
{
 public:
  explicit ItemCompare3(ITraceMng* msg) : m_msg(msg) {}
 public:
  ITraceMng* m_msg;
  ItemInternalArrayView m_items;
 public:
  bool operator()(Integer item1,Integer item2) const
  {
    m_msg->info() << "** Compare ptr=" << m_items.data()
                  << " i1=" << item1 << " i2=" << item2
                  << " i1=" << m_items[item1] << " i2=" << m_items[item2]
                  << " uid1=" << m_items[item1]->uniqueId()
                  << " uid2=" << m_items[item2]->uniqueId();
    return m_items[item1]->uniqueId() < m_items[item2]->uniqueId();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
_sortConnectedItems(IItemFamily* family, IncrementalItemConnectivity* connectivity)
{
  if (!connectivity)
    return;

  // Trie les entités connectées aux noeuds par uniqueId() croissant.
  // Cela est utile pour garantir un ordre de parcours de ces entités
  // identique quel que soit le découpage et ainsi améliorer
  // la reproductibilité.
  ItemInfoListView items_infos(family->itemInfoListView());
  ItemCompare2 ic_items(items_infos);
  ENUMERATE_ITEM(iitem,allItems()){
    ItemLocalId lid(iitem.itemLocalId());
    Int32ArrayView conn_lids = connectivity->_connectedItemsLocalId(lid);
    std::sort(std::begin(conn_lids),std::end(conn_lids),ic_items);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
sortInternalReferences()
{
  // Trie les mailles connectées aux noeuds par uniqueId() croissant.
  _sortConnectedItems(mesh()->cellFamily(),m_cell_connectivity->trueCustomConnectivity());

  // Fait de même pour les faces et les arêtes.
  // Pour des raisons historiques, cela n'est pas actif par défaut.
  bool do_sort = properties()->getBoolWithDefault("sort-connected-faces-edges",m_is_sort_connected_faces_and_edges);
  if (do_sort){
    info(4) << "Sorting connected faces and edges family=" << fullName();
    _sortConnectedItems(m_face_family,m_face_connectivity->trueCustomConnectivity());
    if (Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_NodeToEdge))
      _sortConnectedItems(mesh()->edgeFamily(),m_edge_connectivity->trueCustomConnectivity());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
notifyItemsUniqueIdChanged()
{
  ItemFamily::notifyItemsUniqueIdChanged();
  // Si les uniqueId() des noeuds changent, cela peut avoir une influence sur
  // l'orientation des faces et des arêtes. Il faut donc renuméroter ces dernières
  m_face_family->reorientFacesIfNeeded();
  m_edge_family->reorientEdgesIfNeeded();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
_addParentNodeToNode(Node parent_node, Node child_node)
{
  m_hparent_connectivity->addConnectedItem(ItemLocalId(child_node), ItemLocalId(parent_node));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeFamily::
_addChildNodeToNode(Node parent_node, Node child_node)
{
  m_hchild_connectivity->addConnectedItem(ItemLocalId(parent_node), ItemLocalId(child_node));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
