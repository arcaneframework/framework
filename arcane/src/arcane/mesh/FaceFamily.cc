// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceFamily.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Famille de faces.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/FaceFamily.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/String.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
#include "arcane/mesh/NewWithLegacyConnectivity.h"
#include "arcane/mesh/FaceReorienter.h"

#include "arcane/IMesh.h"
#include "arcane/ITiedInterface.h"
#include "arcane/TiedFace.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Connectivity.h"

//! AMR
#include "arcane/GeometricUtilities.h"
#include "arcane/SharedVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FaceFamily::TopologyModifier
: public AbstractItemFamilyTopologyModifier
{
 public:

  explicit TopologyModifier(FaceFamily* f)
  :  AbstractItemFamilyTopologyModifier(f), m_true_family(f){}
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
  void replaceCell(ItemLocalId item_lid,Integer index,ItemLocalId new_lid) override
  {
    m_true_family->replaceCell(item_lid,index,new_lid);
  }
  void setBackAndFrontCells(FaceLocalId face_lid, CellLocalId back_cell_lid,
                            CellLocalId front_cell_lid) override
  {
    Face face(m_true_family->_itemInternal(face_lid));
    m_true_family->setBackAndFrontCells(face, back_cell_lid, front_cell_lid);
  }

 private:

  FaceFamily* m_true_family = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceFamily::
FaceFamily(IMesh* mesh,const String& name)
: ItemFamily(mesh,IK_Face,name)
{
  _setTopologyModifier(new TopologyModifier(this));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceFamily::
~FaceFamily()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
build()
{
  ItemFamily::build();

  m_node_family = ARCANE_CHECK_POINTER(dynamic_cast<NodeFamily*>(m_mesh->nodeFamily()));
  m_edge_family = ARCANE_CHECK_POINTER(dynamic_cast<EdgeFamily*>(m_mesh->edgeFamily()));

  if (m_mesh->useMeshItemFamilyDependencies()) // temporary to fill legacy, even with family dependencies
  {
    m_node_connectivity = dynamic_cast<NewWithLegacyConnectivityType<FaceFamily,NodeFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->nodeFamily(),connectivityName(this,mesh()->nodeFamily())));
    m_edge_connectivity = dynamic_cast<NewWithLegacyConnectivityType<FaceFamily,EdgeFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->edgeFamily(),connectivityName(this,mesh()->edgeFamily())));
    m_face_connectivity = dynamic_cast<NewWithLegacyConnectivityType<FaceFamily,FaceFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->faceFamily(),connectivityName(this,mesh()->faceFamily())));
    m_cell_connectivity = dynamic_cast<NewWithLegacyConnectivityType<FaceFamily,CellFamily>::type*>(m_mesh->itemFamilyNetwork()->getConnectivity(this,mesh()->cellFamily(),connectivityName(this,mesh()->cellFamily())));
  }
  else
  {
    m_node_connectivity = new NodeConnectivity(this,mesh()->nodeFamily(),"FaceNode");
    m_edge_connectivity = new EdgeConnectivity(this,mesh()->edgeFamily(),"FaceEdge");
    m_face_connectivity = new FaceConnectivity(this,mesh()->faceFamily(),"FaceFace");
    m_cell_connectivity = new CellConnectivity(this,mesh()->cellFamily(),"FaceCell");
  }
  m_hparent_connectivity = new HParentConnectivity(this, this, "HParentFace");
  m_hchild_connectivity = new HChildConnectivity(this, this, "HChildFace");

  _addConnectivitySelector(m_node_connectivity);
  _addConnectivitySelector(m_edge_connectivity);
  _addConnectivitySelector(m_face_connectivity);
  _addConnectivitySelector(m_cell_connectivity);
  _addConnectivitySelector(m_hparent_connectivity);
  _addConnectivitySelector(m_hchild_connectivity);

  _buildConnectivitySelectors();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void FaceFamily::
_createOne(ItemInternal* item,Int64 uid,ItemTypeInfo* type)
{
  m_item_internal_list->faces = _itemsInternal();
  _allocateInfos(item,uid,type);
  auto nc = m_node_connectivity->trueCustomConnectivity();
  if (nc)
    nc->addConnectedItems(ItemLocalId(item),type->nbLocalNode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void FaceFamily::
_createOne(ItemInternal* item,Int64 uid,ItemTypeId type_id)
{
  _createOne(item,uid,_itemTypeMng()->typeFromId(type_id));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Alloue une face de numéro unique \a uid et de type \a type. Ajout générique d'item.
 *
 * Cette version est faite pour être appelée dans un bloc générique ignorant le type
 * de l'item. La mise à jour du nombre d'item du maillage est donc fait dans cette méthode,
 * et non dans le bloc appelant.
 */
Item FaceFamily::
allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info)
{
  ++mesh_info.nbFace();
  return allocOne(uid,type_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Alloue une face de numéro unique \a uid et de type \a type.
 */
ItemInternal* FaceFamily::
allocOne(Int64 uid,ItemTypeInfo* type)
{
  ItemInternal* item = _allocOne(uid);
  _createOne(item,uid,type);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère ou alloue une face de numéro unique \a uid et de type \a type.
 *
 * Cette version est faite pour être appelée dans un bloc générique ignorant le type
 * de l'item. La mise à jour du nombre d'item du maillage est donc fait dans cette méthode,
 * et non dans le bloc appelant.
 * Si une face de numéro unique \a uid existe déjà, la retourne. Sinon,
 * la face est créée. \a is_alloc est vrai si la face vient d'être créée.
 *
 */
Item FaceFamily::
findOrAllocOne(Int64 uid,ItemTypeId type_id,MeshInfos& mesh_info, bool& is_alloc)
{
  auto face = findOrAllocOne(uid,type_id,is_alloc);
  if (is_alloc)
    ++mesh_info.nbFace();
  return face;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère ou alloue une face de numéro unique \a uid et de type \a type.
 *
 * Si une face de numéro unique \a uid existe déjà, la retourne. Sinon,
 * la face est créée. \a is_alloc est vrai si la face vient d'être créée.
 */
ItemInternal* FaceFamily::
findOrAllocOne(Int64 uid,ItemTypeInfo* type,bool& is_alloc)
{
  ItemInternal* item = _findOrAllocOne(uid,is_alloc);
  if (is_alloc){
    _createOne(item,uid,type);
  }
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Face FaceFamily::
allocOne(Int64 uid,ItemTypeId type_id)
{
  ItemInternal* item = _allocOne(uid);
  _createOne(item,uid,type_id);
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Face FaceFamily::
findOrAllocOne(Int64 uid,ItemTypeId type_id,bool& is_alloc)
{
  ItemInternal* item = _findOrAllocOne(uid,is_alloc);
  if (is_alloc){
    _createOne(item,uid,type_id);
  }
  return item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
preAllocate(Integer nb_item)
{
  if (!m_has_face)
    return;
  // On ne préalloue pas par défaut car on ne souhaite pas pré-allouer
  // pour la connectivité face->face car c'est dernière n'est utilisée que
  // s'il y a des TiedInterface et cela n'est pas fréquent.
  // Du coup on pré-alloue explicitement les connectivités qu'on souhaite.
  this->_preAllocate(nb_item,false);
  m_node_connectivity->trueCustomConnectivity()->reserveMemoryForNbSourceItems(nb_item,true);
  if (Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasEdge))
    m_edge_connectivity->trueCustomConnectivity()->reserveMemoryForNbSourceItems(nb_item,true);
  m_cell_connectivity->trueCustomConnectivity()->reserveMemoryForNbSourceItems(nb_item,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
computeSynchronizeInfos()
{
  debug() << "Creating the list of ghosts faces";
  ItemFamily::computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace le noeud d'index \a index de la face \a face avec
 * celui de localId() \a node_lid.
 */
void FaceFamily::
replaceNode(ItemLocalId face,Integer index,ItemLocalId node)
{
  m_node_connectivity->replaceItem(face,index,node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace l'arête d'index \a index de la face \a face avec
 * celle de localId() \a edge_lid.
 */
void FaceFamily::
replaceEdge(ItemLocalId face,Integer index,ItemLocalId edge)
{
  m_edge_connectivity->replaceItem(face,index,edge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace la face d'index \a index de la face \a face avec
 * celle de localId() \a face_lid.
 */
void FaceFamily::
replaceFace(ItemLocalId face,Integer index,ItemLocalId face2)
{
  m_face_connectivity->replaceItem(face,index,face2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplace la maille d'index \a index de la face \a face avec
 * celle de localId() \a cell_lid.
 */
void FaceFamily::
replaceCell(ItemLocalId face,Integer index,ItemLocalId cell)
{
  m_cell_connectivity->replaceItem(face,index,cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne la maille derrière et devant la face.
 *
 * \a iback_cell_lid ou \a ifront_cell_lid peuvent valoir NULL_ITEM_LOCAL_ID
 * pour indiquer qu'il n'y a pas de maille derrière ou devant. Si les deux
 * valeurs sont NULL_ITEM_LOCAL_ID cela signifie que la face n'a pas de
 * mailles connectées.
 */
void FaceFamily::
setBackAndFrontCells(Face face,Int32 iback_cell_lid,Int32 ifront_cell_lid)
{
  face.mutableItemBase()._setFaceBackAndFrontCells(iback_cell_lid,ifront_cell_lid);
  ItemLocalId back_cell_lid(iback_cell_lid);
  ItemLocalId front_cell_lid(ifront_cell_lid);
  auto c = m_cell_connectivity->trueCustomConnectivity();
  if (c){
    ItemLocalId face_lid(face.localId());
    // Supprime toutes les mailles connectées.
    // TODO: optimiser en ne supprimant pas s'il n'y a pas besoin pour éviter
    // des réallocations.
    c->removeConnectedItems(face_lid);
    if (front_cell_lid==NULL_ITEM_LOCAL_ID){
      if (back_cell_lid!=NULL_ITEM_LOCAL_ID){
        // Reste uniquement la back_cell ou aucune maille.
        c->IncrementalItemConnectivity::addConnectedItem(face_lid,back_cell_lid); // add the class name for the case of Face to Cell connectivity. The class is overridden to handle family dependencies but the base method must be called here.
      }
      // Ici reste aucune maille mais comme on a tout supprimé il n'y a rien
      // à faire.
    }
    else if (back_cell_lid==NULL_ITEM_LOCAL_ID){
      // Reste uniquement la front cell
      c->IncrementalItemConnectivity::addConnectedItem(face_lid,front_cell_lid);
    }
    else{
      // Il y a deux mailles connectées. La back_cell est toujours la première.
      c->IncrementalItemConnectivity::addConnectedItem(face_lid,back_cell_lid);
      c->IncrementalItemConnectivity::addConnectedItem(face_lid,front_cell_lid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
addBackCellToFace(Face face,Cell new_cell)
{
  _checkValidSourceTargetItems(face,new_cell);

  Integer nb_cell = face.nbCell();

  // SDP: les tests suivants sont imcompatibles avec le raffinement
  // par couches
  if (m_check_orientation){
    Cell current_cell = face.backCell();
    if (face.hasFlags(ItemFlags::II_HasBackCell)){
      ARCANE_FATAL("Face already having a back cell."
                   " This is most probably due to the fact that the face"
                   " is connected to a reverse cell with a negative volume."
                   " Face={0}. new_cell={1} current_cell={2}",FullItemPrinter(face),
                   FullItemPrinter(new_cell),FullItemPrinter(current_cell));
    }
  }

  if (nb_cell>=2)
    ARCANE_FATAL("face '{0}' already has two cells",FullItemPrinter(face));

  _updateSharedInfo();

  // Si on a déjà une maille, il s'agit de la front cell.
  Int32 front_cell_lid = (nb_cell==1) ? face.cellId(0) : NULL_ITEM_LOCAL_ID;
  setBackAndFrontCells(face,new_cell.localId(),front_cell_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
addFrontCellToFace(Face face,Cell new_cell)
{
  _checkValidSourceTargetItems(face,new_cell);

  Integer nb_cell = face.nbCell();

  // SDP: les tests suivants sont imcompatibles avec le raffinement
  // par couches
  if (m_check_orientation){
    Cell current_cell = face.frontCell();
    if (face.hasFlags(ItemFlags::II_HasFrontCell)){
      ARCANE_FATAL("Face already having a front cell."
                   " This is most probably due to the fact that the face"
                   " is connected to a reverse cell with a negative volume."
                   " Face={0}. new_cell={1} current_cell={2}",FullItemPrinter(face),
                   FullItemPrinter(new_cell),FullItemPrinter(current_cell));
    }
  }

  if (nb_cell>=2)
    ARCANE_FATAL("face '{0}' already has two cells",FullItemPrinter(face));

  _updateSharedInfo();

  // Si on a déjà une maille, il s'agit de la back cell.
  Int32 back_cell_lid = (nb_cell==1) ? face.cellId(0) : NULL_ITEM_LOCAL_ID;
  setBackAndFrontCells(face,back_cell_lid,new_cell.localId());
}

//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
replaceBackCellToFace(Face face,ItemLocalId new_cell)
{
  ARCANE_ASSERT((face.nbCell() ==2),("Face should have back and front cells"));

  Cell current_cell = face.backCell();
  _topologyModifier()->findAndReplaceCell(face,current_cell,new_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
replaceFrontCellToFace(Face face,ItemLocalId new_cell)
{
  ARCANE_ASSERT((face.nbCell() ==2),("Face should have back and front cells"));

  Cell current_cell = face.frontCell();
  _topologyModifier()->findAndReplaceCell(face,current_cell,new_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
addBackFrontCellsFromParentFace(Face subface,Face face)
{
	Cell fcell= face.frontCell();
	Cell bcell= face.backCell();

	if (subface.hasFlags(ItemFlags::II_HasBackCell)){
    if(fcell.isActive()) 
      addFrontCellToFace(subface,face.frontCell());
    else if(bcell.isActive())
      addFrontCellToFace(subface,face.backCell());
	}
	else if (subface.hasFlags(ItemFlags::II_HasFrontCell)){
    if(bcell.isActive()) 
      addBackCellToFace(subface,face.backCell());
    else if (fcell.isActive())
      addBackCellToFace(subface,face.frontCell());
	}
	ARCANE_ASSERT((subface.backCell() != subface.frontCell()),("back front cells error"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
replaceBackFrontCellsFromParentFace(Cell subcell,Face subface,
                                    Cell cell,Face face)
{
  Cell fscell= subface.frontCell();
  Cell bscell= subface.backCell();
  Cell fcell= face.frontCell();
  Cell bcell= face.backCell();
  if (fscell.localId()==subcell.localId()) {
    if(fcell.localId()==cell.localId()) {
      if(bcell.level()>bscell.level()) {
        replaceBackCellToFace(subface,bcell);
      }
    }
    else {
      if(fcell.level() > bscell.level()){
        replaceBackCellToFace(subface,fcell);
      }
    }
  }
  else {
    if (fcell.localId()==cell.localId()){
      if (bcell.level() > fscell.level()){
        replaceFrontCellToFace(subface,bcell);
      }
    }
    else{
      if(fcell.level()>fscell.level()){
        replaceFrontCellToFace(subface,fcell);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceFamily::
isSubFaceInFace(Face subface,Face face) const
{
  const Real tol= 1e-02;
  bool is_true= false;

  //TODO : un std::binary_search pour accelerer la recherche
  for( Node inode : subface.nodes() ){
    for( Node inode2 : face.nodes() ){
      if (inode.uniqueId() == inode2.uniqueId()) {// il suffit qu'un seul noeud concide
        is_true = true;
        break;
      }
    }
  }
  if (!is_true)
    return false;

  SharedVariableNodeReal3 orig_nodes_coords(mesh()->sharedNodesCoordinates());
  Real3 normal_face = _computeFaceNormal(face,orig_nodes_coords);
  Real3 normal_subface = _computeFaceNormal(subface,orig_nodes_coords);
  Real ps = math::dot(normal_face,normal_subface);
  Real residual = math::abs(ps)-1.;
  return math::abs(residual) < tol ? true: false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceFamily::
isChildOnFace(ItemWithNodes child,Face face) const
{
	//TODO : un std::binary_search pour accelerer la recherche
	for( Node inode : face.nodes() ){
		for( Node inode2 : child.nodes() ){
			if (inode.uniqueId() == inode2.uniqueId()) // il suffit qu'un seul noeud concide
				return true;
		}
	}
	return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
subFaces(Face face,Array<ItemInternal*>& subfaces)
{
	Cell cell;
	const Integer nb_cell= face.nbCell();

	if (nb_cell == 2) {
		Cell fcell = face.frontCell();
		cell = (fcell.hasHChildren()) ? face.frontCell() : face.backCell();
	}
	else
		cell = face.cell(0);

	for( Integer c=0;c<cell.nbHChildren();c++){
		Cell child = cell.hChild(c);
		if (isChildOnFace(child,face)){
			//debug() << "current FACE:" << FullItemPrinter(face)
			//<< "\n current child CELL:" << FullItemPrinter(child)
			//<< "\n";
			for( Face subface : child.faces() ){
				if(isSubFaceInFace(subface,face)){
          subfaces.add(ItemCompatibility::_itemInternal(subface));
				}
			}
		}
	}
	//info() << "\n SUBFACE NB= " << subfaces.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
allSubFaces(Face face,Array<ItemInternal*>& subfaces)
{
  subfaces.add(ItemCompatibility::_itemInternal(face));

  Cell cell;
  const Integer nb_cell= face.nbCell();
  if (nb_cell == 2){
    Cell fcell= face.frontCell();
    if(fcell.hasHChildren() && fcell.isOwn())
      cell =  face.frontCell() ;
    else
      cell = face.backCell();
  }
  else{
    cell= face.cell(0);
  }

  for(Integer c=0;c<cell.nbHChildren();c++){
    Cell child = cell.hChild(c);
    if (isChildOnFace(child,face)){
      for( Face subface : child.faces() ){
        if(isSubFaceInFace(subface,face)){
          allSubFaces(subface,subfaces);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
activeSubFaces(Face face,Array<ItemInternal*>& subfaces)
{
	Cell cell;
	const Integer nb_cell= face.nbCell();

	if(nb_cell==2) {
		Cell fcell = face.frontCell();
		Cell bcell = face.backCell();
		if(fcell.level() > bcell.level())
			cell = face.frontCell();
		else {
			if(bcell.isOwn())
				cell= face.backCell();
			else
				cell= face.frontCell();
		}
	}
	else {
		cell = face.cell(0);
	}
	Cell pcell = cell.topHParent();

	UniqueArray<ItemInternal*> cell_family;
	activeFamilyTree (cell_family,pcell);
	for(Integer c=0;c<cell_family.size();c++){
		Cell child = cell_family[c];
		if(isChildOnFace(child,face)){
			//debug() << "current FACE:" << FullItemPrinter(face)
			//<< "\n current child CELL:" << FullItemPrinter(child)
			//<< "\n";
			for( Face subface : child.faces() ){
				if(isSubFaceInFace(subface,face)){
          subfaces.add(ItemCompatibility::_itemInternal(subface));
				}
			}
		}
	}
	//info() << "\n SUBFACE NB= " << subfaces.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
familyTree(Array<ItemInternal*>& family,Cell item,
           const bool reset) const
{
	ARCANE_ASSERT((!item.isSubactive()),("The family tree doesn't include subactive items"));
	// Clear the array if the flag reset tells us to.
	if (reset)
		family.clear();
	// Add this item to the family tree.
	family.add(ItemCompatibility::_itemInternal(item));
	// Recurse into the items children, if it has them.
	// Do not clear the array any more.
	if (!item.isActive())
		for (Integer c=0, cs=item.nbHChildren(); c<cs; c++){
			Cell ichild= item.hChild(c);
      familyTree (family,ichild, false);
		}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
activeFamilyTree(Array<ItemInternal*>& family,Cell item,const bool reset) const
{
	ARCANE_ASSERT((!item.isSubactive()),("The family tree doesn't include subactive items"));
	// Clear the array if the flag reset tells us to.
	if (reset)
		family.clear();
	// Add this item to the family tree.
	if(item.isActive())
		family.add(ItemCompatibility::_itemInternal(item));
	else
		for (Integer c=0, cs=item.nbHChildren(); c<cs; c++){
			Cell ichild= item.hChild(c);
			if (ichild.isOwn())
				activeFamilyTree(family,ichild,false);
		}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_addParentFaceToFace(Face parent_face, Face child_face)
{
  m_hparent_connectivity->addConnectedItem(ItemLocalId(child_face), ItemLocalId(parent_face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_addChildFaceToFace(Face parent_face, Face child_face)
{
  m_hchild_connectivity->addConnectedItem(ItemLocalId(parent_face), ItemLocalId(child_face));
}

// OFF AMR

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
addEdgeToFace(Face face,Edge new_edge)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_FaceToEdge))
    return;

  _checkValidSourceTargetItems(face,new_edge);
  m_edge_connectivity->addConnectedItem(ItemLocalId(face),ItemLocalId(new_edge));;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
removeEdgeFromFace(Face face,Edge edge_to_remove)
{
  if (!Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_FaceToEdge))
    return;

  _checkValidSourceTargetItems(face,edge_to_remove);
  m_edge_connectivity->removeConnectedItem(ItemLocalId(face),ItemLocalId(edge_to_remove));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void FaceFamily::
_removeFace(Face face)
{
  ItemLocalId face_lid = face;
  for( Int32 edge : face.edgeIds() )
    m_edge_family->removeFaceFromEdge(ItemLocalId(edge),face_lid);
  Int32 nb_linear_node = face.nbLinearNode();
  for( Int32 i=0; i<nb_linear_node; ++i ){
    m_node_family->removeFaceFromNode(face.nodeId(i),face_lid);
  }

  _removeOne(face);
  // On ne supprime pas ici les autres relations (ici aucune)
  // Car l'autre de suppression doit toujours être cell, face, edge, node
  // donc node est en dernier et tout est déjà fait
  // Par ailleurs, cela évite des problèmes de récursivité
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 FaceFamily::
_computeFaceNormal(Face face, const SharedVariableNodeReal3& nodes_coord) const
{
  Real3 normal_face = Real3::zero();
  Integer nb_node= face.nbNode();
  Real3UniqueArray cord_face(nb_node);
  for( Integer i=0; i<nb_node; ++i ){
    cord_face[i] = nodes_coord[face.node(i)];
  }
  switch(nb_node)
  {
    case(4):
      {
        GeometricUtilities::QuadMapping face_mapping;
        face_mapping.m_pos[0] = cord_face[0];
        face_mapping.m_pos[1] = cord_face[1];
        face_mapping.m_pos[2] = cord_face[2];
        face_mapping.m_pos[3] = cord_face[3];
        normal_face= face_mapping.normal();
      }
      break ;
    case(3):
      {
        Real3 v1 = cord_face[1] - cord_face[0];
        Real3 v2 = cord_face[2] - cord_face[0];
        normal_face = math::vecMul(v1,v2).normalize();
      }
      break ;
    case(2):
      {
      normal_face=cord_face[0]-cord_face[1];
      normal_face.normalize();
      }
    break ;
    default: ARCANE_FATAL("This kind of face is not handled");
  }
  return normal_face;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
removeCellFromFace(Face face,ItemLocalId cell_to_remove_lid)
{
  _checkValidItem(face);

  Integer nb_cell = face.nbCell();

#ifdef ARCANE_CHECK
  if (face.itemBase().isSuppressed())
    ARCANE_FATAL("Can not remove cell from destroyed face={0}",ItemPrinter(face));
  if (nb_cell==0)
    ARCANE_FATAL("Can not remove cell lid={0} from face uid={1} with no cell connected",
                 cell_to_remove_lid, face.uniqueId());
#endif /* ARCANE_CHECK */

  Integer nb_cell_after = nb_cell-1;
  const Int32 null_cell_lid = NULL_ITEM_LOCAL_ID;
  //! AMR
  // forcer la suppression d'une face entre deux mailles de niveaux differents
  // car cette face n'est topologiquement attachée a la maille de niveau inferieur
  if (nb_cell == 2){
    if(face.backCell().level() != face.frontCell().level())
	  nb_cell_after = 0;
    else if (! (face.backCell().isActive() && face.frontCell().isActive())){
      // TODO: GG: pour des raisons de performance, il est préférable d'éviter
      // les allocations dans cette méthode car elle est appelée très souvent.
      UniqueArray<ItemInternal*> subfaces;
      subFaces(face,subfaces);
      for(Integer s=0,ss=subfaces.size();s<ss;s++){
        ItemInternal* face2= subfaces[s];
        Int32 cell0 = face2->cellId(0);
        Int32 cell1 = face2->cellId(1);
        // On avait obligatoirement deux mailles connectées avant,
        // donc la back_cell est la maille 0, la front cell la maille 1
        if (cell0==cell_to_remove_lid){
          // Reste la front cell
          setBackAndFrontCells(face,null_cell_lid,cell1);
        }
        else{
          // Reste la back cell
          setBackAndFrontCells(face,cell0,null_cell_lid);
        }
        _updateSharedInfo();
      }
    }
  }
  // OFF AMR
  if (nb_cell_after!=0){
    Int32 cell0 = face.cellId(0);
    Int32 cell1 = face.cellId(1);
    // On avait obligatoirement deux mailles connectées avant,
    // donc la back_cell est la maille 0, la front cell la maille 1
    if (cell0==cell_to_remove_lid){
      // Reste la front cell
      setBackAndFrontCells(face,null_cell_lid,cell1);
    }
    else{
      // Reste la back cell
      setBackAndFrontCells(face,cell0,null_cell_lid);
    }
  }
  else{
    setBackAndFrontCells(face,null_cell_lid,null_cell_lid);
  }

  _updateSharedInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
removeFaceIfNotConnected(Face face)
{
	_checkValidItem(face);

	if (!face.itemBase().isSuppressed() && face.nbCell()==0){
    _removeFace(face);
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_addMasterFaceToFace(Face face,Face master_face)
{
  m_face_connectivity->addConnectedItem(face,master_face);
  face.mutableItemBase().addFlags(ItemFlags::II_SlaveFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_addSlaveFacesToFace(Face master_face,Int32ConstArrayView slave_faces_lid)
{
  Integer nb_slave = slave_faces_lid.size();
  for( Integer i=0; i<nb_slave; ++i )
    m_face_connectivity->addConnectedItem(master_face,ItemLocalId(slave_faces_lid[i]));
  master_face.mutableItemBase().addFlags(ItemFlags::II_MasterFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_removeMasterFaceToFace(Face face)
{
  m_face_connectivity->removeConnectedItems(face);
  face.mutableItemBase().removeFlags(ItemFlags::II_SlaveFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
_removeSlaveFacesToFace(Face master_face)
{
  m_face_connectivity->removeConnectedItems(master_face);
  master_face.mutableItemBase().removeFlags(ItemFlags::II_MasterFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
applyTiedInterface(ITiedInterface* interface)
{
  TiedInterfaceFaceList tied_faces(interface->tiedFaces());
  Int32UniqueArray slave_faces;

  ENUMERATE_FACE(imaster_face,interface->masterInterface()){
    Face master_face = *imaster_face;
    Integer index = imaster_face.index();
    ConstArrayView<TiedFace> slave_tied_faces(tied_faces[index]);
    Integer nb_slave = slave_tied_faces.size();
    slave_faces.clear();
    for( Integer zz=0; zz<nb_slave; ++zz ){
      TiedFace tn = tied_faces[index][zz];
      Face slave_face = tn.face();
      slave_faces.add(slave_face.localId());
      _addMasterFaceToFace(slave_face,master_face);
    }
    _addSlaveFacesToFace(master_face,slave_faces);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
removeTiedInterface(ITiedInterface* interface)
{
  TiedInterfaceFaceList tied_faces(interface->tiedFaces());

  ENUMERATE_FACE(imaster_face,interface->masterInterface()){
    Face master_face = *imaster_face;
    Integer index = imaster_face.index();
    ConstArrayView<TiedFace> slave_tied_faces(tied_faces[index]);
    Integer nb_slave = slave_tied_faces.size();
    for( Integer zz=0; zz<nb_slave; ++zz ){
      TiedFace tn = tied_faces[index][zz];
      Face slave_face = tn.face();
      _removeMasterFaceToFace(slave_face);
    }
    _removeSlaveFacesToFace(master_face);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
setConnectivity(const Integer c)
{
  m_mesh_connectivity = c;
  m_has_face = Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasFace);
  if (m_has_face){
    m_node_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Face,IK_Node);
    if (Connectivity::hasConnectivity(m_mesh_connectivity,Connectivity::CT_HasEdge))
      m_edge_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Face,IK_Edge);
    m_face_connectivity->setPreAllocatedSize(4);
    m_cell_prealloc = Connectivity::getPrealloc(m_mesh_connectivity,IK_Face,IK_Cell);
    m_node_connectivity->setPreAllocatedSize(m_node_prealloc);
    m_cell_connectivity->setPreAllocatedSize(m_cell_prealloc);
    debug() << "Family " << name() << " prealloc "
            << m_node_prealloc << " by node, "
            << m_edge_prealloc << " by edge, "
            << m_cell_prealloc << " by cell.";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceFamily::
reorientFacesIfNeeded()
{
  // Réoriente les faces si nécessaire. Cela est le cas par exemple si on
  // a changé la numérotation des uniqueId() des noeuds.
  mesh::FaceReorienter face_reorienter(mesh());
  ENUMERATE_ (Face, iface, allItems()) {
    face_reorienter.checkAndChangeOrientationAMR(*iface);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
