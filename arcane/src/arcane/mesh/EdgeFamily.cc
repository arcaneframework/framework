// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeFamily.cc                                               (C) 2000-2017 */
/*                                                                           */
/* Famille d'arêtes.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/IMesh.h"

#include "arcane/MeshUtils.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Connectivity.h"

#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
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
, m_has_edge(false)
, m_node_prealloc(0)
, m_face_prealloc(0)
, m_cell_prealloc(0)
, m_mesh_connectivity(0)
, m_node_connectivity(nullptr)
, m_face_connectivity(nullptr)
, m_cell_connectivity(nullptr)
, m_node_family(nullptr)
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
ItemInternal* EdgeFamily::
allocOne(Int64 uid,ItemTypeInfo* type, MeshInfos& mesh_info)
{
  ARCANE_ASSERT((type->typeId() == IT_Line2),(""));
  ARCANE_UNUSED(type);
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
ItemInternal* EdgeFamily::
findOrAllocOne(Int64 uid,ItemTypeInfo* type,MeshInfos& mesh_info, bool& is_alloc)
{
  ARCANE_ASSERT((type->typeId() == IT_Line2),(""));
  ARCANE_UNUSED(type);
  auto edge = findOrAllocOne(uid,is_alloc);
  if (is_alloc) ++mesh_info.nbEdge();
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
  Integer mem = 0;
  Integer base_mem = ItemSharedInfo::COMMON_BASE_MEMORY;
  if (m_has_edge) { // On n'alloue rien du tout si on n'a pas d'arête
    mem = base_mem * (nb_item+1);
  }
  _reserveInfosMemory(mem);
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
addCellToEdge(ItemInternal* edge,ItemInternal* new_cell)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToCell))
    return;
  _checkValidSourceTargetItems(edge,new_cell);
  m_cell_connectivity->addConnectedItem(ItemLocalId(edge),ItemLocalId(new_cell));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
addFaceToEdge(ItemInternal* edge,ItemInternal* new_face)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToFace))
    return;
  _checkValidSourceTargetItems(edge,new_face);
  m_face_connectivity->addConnectedItem(ItemLocalId(edge),ItemLocalId(new_face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void EdgeFamily::
_removeEdge(ItemInternal* iedge)
{
  Edge edge(iedge);
  for( Node node : edge.nodes() )
    m_node_family->removeEdgeFromNode(node,edge);
  _removeOne(iedge);
  // On ne supprime pas ici les autres relations (face->edge,cell->edge)
  // Car l'ordre de suppression doit toujours être cell, face, edge, node
  // donc node est en dernier et tout est déjà fait
  // Par ailleurs, cela évite des problèmes de récursivité
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeCellFromEdge(ItemInternal* edge,ItemInternal* cell_to_remove, bool no_destroy)
{
  _checkValidItem(cell_to_remove);
  if (!no_destroy)
    throw NotSupportedException(A_FUNCINFO,"no_destroy==false");
  removeCellFromEdge(edge,ItemLocalId(cell_to_remove));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeCellFromEdge(ItemInternal* edge,ItemLocalId cell_to_remove_lid)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToCell))
    return;
  _checkValidItem(edge);
  m_cell_connectivity->removeConnectedItem(ItemLocalId(edge),cell_to_remove_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeFamily::
removeFaceFromEdge(ItemInternal* edge,ItemInternal* face_to_remove)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_EdgeToFace))
    return;
  _checkValidSourceTargetItems(edge,face_to_remove);
  m_face_connectivity->removeConnectedItem(ItemLocalId(edge),ItemLocalId(face_to_remove));
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
removeEdgeIfNotConnected(ItemInternal* edge)
{
	_checkValidItem(edge);
	if (!edge->isSuppressed() && edge->nbCell()==0){
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

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
