// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshChecker.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Classe fournissant des méthodes de vérification sur le maillage.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/ItemGroupsSynchronize.h"
#include "arcane/mesh/FaceReorienter.h"
#include "arcane/mesh/DynamicMeshChecker.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/TemporaryVariableBuildInfo.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/NodesOfItemReorderer.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshChecker::
DynamicMeshChecker(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
  if (arcaneIsCheck())
    m_check_level = 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshChecker::
~DynamicMeshChecker()
{
  delete m_var_cells_faces;
  delete m_var_cells_nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérification sommaire de la validité du maillage.
 * Les vérifications portent sur les points suivants:
 * - pas d'entités du maillage ayant un indice nul.
 * - pour les faces, vérifie qu'il existe au moins une frontCell ou une
 * backCell.
 * Une erreur fatale est générée en cas de non respect de ces règles.
 */
void DynamicMeshChecker::
checkValidMesh()
{
  info(4) << "Check mesh coherence.";
  checkValidConnectivity();

  for( IItemFamilyCollection::Enumerator i(m_mesh->itemFamilies()); ++i; ){
    IItemFamily* family = *i;
    //GG: regarder pourquoi il y a ce test.
    if (family->itemKind()>=NB_ITEM_KIND)
      continue;
    family->checkValid();
  }
  mesh_utils::checkMeshProperties(m_mesh,false,false,true);

  // Sauve dans une variable aux mailles la liste des faces et
  // noeuds qui la composent.
  // Ceci n'est utile que pour des vérifications
  // NOTE: pour l'instant mis a false car fait planter si utilise
  // partitionnement initial dans Arcane
  const bool global_check = false;
  if (m_check_level>=1 && global_check){
    if (!m_var_cells_faces)
      m_var_cells_faces = new VariableCellArrayInt64(VariableBuildInfo(m_mesh,"MeshTestCellFaces"));

    if (!m_var_cells_nodes)
      m_var_cells_nodes = new VariableCellArrayInt64(VariableBuildInfo(m_mesh,"MeshTestCellNodes"));

    CellGroup all_cells(m_mesh->allCells());
    Integer max_nb_face = 0;
    Integer max_nb_node = 0;
    ENUMERATE_CELL(i,all_cells){
      Cell cell = *i;
      Integer nb_face = cell.nbFace();
      Integer nb_node = cell.nbNode();
      if (nb_face>max_nb_face)
        max_nb_face = nb_face;
      if (nb_node>max_nb_node)
        max_nb_node = nb_node;
    }
    m_var_cells_faces->resize(max_nb_face);
    m_var_cells_nodes->resize(max_nb_node);
    ENUMERATE_CELL(i,all_cells){
      Cell cell = *i;
      Integer nb_face = cell.nbFace();
      for( Integer z=0; z<nb_face; ++z )
        (*m_var_cells_faces)[cell][z] = cell.face(z).uniqueId().asInt64();
      Integer nb_node = cell.nbNode();
      for( Integer z=0; z<nb_node; ++z )
        (*m_var_cells_nodes)[cell][z] = cell.node(z).uniqueId().asInt64();
    }
  }
  checkValidMeshFull();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
checkValidMeshFull()
{
  String func_name("DynamicMesh::checkValidMeshFull");
  info() << func_name << " on " << m_mesh->name();
  Integer nb_error = 0;
  VariableFaceInt64 var_faces(VariableBuildInfo(m_mesh,"MeshCheckFaces"));
  debug() << " VAR_FACE GROUP=" << var_faces.variable()->itemFamily()
          << " NAME=" << var_faces.itemGroup().name()
          << " FAMILY_NAME=" << var_faces.variable()->itemFamilyName()
          << " GROUP_NAME=" << var_faces.variable()->itemGroupName();
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Cell c = face.backCell();
    Int64 uid = c.null() ? NULL_ITEM_UNIQUE_ID : c.uniqueId();
    var_faces[face] = uid;
  }
  var_faces.synchronize();
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Cell c = face.backCell();
    Int64 uid = c.null() ? NULL_ITEM_UNIQUE_ID : c.uniqueId();
    if (uid!=var_faces[face] && uid!=NULL_ITEM_ID && var_faces[face]!=NULL_ITEM_ID){
      error() << func_name << " bad back cell in face uid=" << face.uniqueId();
      ++nb_error;
    }
  }

  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Cell c = face.frontCell();
    Int64 uid = c.null() ? NULL_ITEM_UNIQUE_ID : c.uniqueId();
    var_faces[face] = uid;
  }
  var_faces.synchronize();
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Cell c = face.frontCell();
    Int64 uid = c.null() ?  NULL_ITEM_UNIQUE_ID : c.uniqueId();
    if (uid!=var_faces[face] && uid!=NULL_ITEM_ID && var_faces[face]!=NULL_ITEM_ID){
      error() << func_name << " bad front cell in face uid=" << face.uniqueId();
      ++nb_error;
    }
  }

  ENUMERATE_CELL(icell,m_mesh->allCells()){
    Cell cell = *icell;
    Integer cell_type = cell.type();
    if (cell_type == IT_Line2) {
      error() << func_name << " bad cell type in face uid=" << cell.uniqueId();
      ++nb_error;
    }
  }

  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Integer face_type = face.type();
    if (face_type == IT_Vertex) {
      error() << func_name << " bad face type in face uid=" << face.uniqueId();
      ++nb_error;
    }
  }

  if (nb_error!=0)
    ARCANE_FATAL("Invalid mesh");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Vérifie que la connectivité est valide.
 *
 * Les vérifications portent sur les points suivants:
 * - pas d'entités du maillage ayant un indice nul.
 * - pour les faces, vérifie qu'il existe au moins une frontCell ou une
 * backCell. De plus, si elle a deux mailles, la backCell doit
 * toujours être la première.
 * - les noeuds et les faces doivent avoir le même propriétaire
 * qu'une des mailles attenante.
 * - vérifie que les faces sont bien ordonnées et orientées
 */
void DynamicMeshChecker::
checkValidConnectivity()
{
  String func_name = "MeshChecker::checkValidConnectivity";
  debug() << func_name << " check";

  // Appelle la méthode de vérification de chaque famille.
  IItemFamilyCollection item_families = m_mesh->itemFamilies();
  for( auto x = item_families.enumerator(); ++x; ){
    IItemFamily* f = *x;
    f->checkValidConnectivity();
  }

  if (!m_mesh->parentMesh()){
    Integer index = 0;
    ENUMERATE_(Face,i,m_mesh->allFaces()){
      Face elem = *i;
      Cell front_cell = elem.frontCell();
      Cell back_cell  = elem.backCell();
      if (front_cell.null() && back_cell.null())
        ARCANE_FATAL("Local face '{0}' has two null cell face=",index,ItemPrinter(elem));
      index++;
    }
  }

  // Vérifie que la connective maille<->noeud
  // est réciproque
  ENUMERATE_NODE(inode,m_mesh->allNodes()){
    Node node = *inode;
    for( Cell cell : node.cells() ){
      bool has_found = false;
      for( Node node2 : cell.nodes() ){
        if (node2==node){
          has_found = true;
          break;
        }
      }
      if (!has_found){
        ARCANE_FATAL("Node uid={0} is connected to the cell uid={1} but the cell"
                     " is not connected to this node.",ItemPrinter(node),ItemPrinter(cell));
      }
    }
  }

  // ATT: les verifications suivantes ne sont pas compatible avec l'AMR
  // TODO : etendre les verifs au cas AMR
  //! AMR
  if(!m_mesh->isAmrActivated()){
    // Vérifie que la connective maille<->arêtes
    // est réciproque
    ENUMERATE_EDGE(iedge,m_mesh->allEdges()){
      Edge edge = *iedge;
      for( Cell cell : edge.cells() ){
        bool has_found = false;
        for( Edge edge2 : cell.edges() ){
          if (edge2==edge){
            has_found = true;
            break;
          }
        }
        if (!has_found){
          ARCANE_FATAL("Edge uid={0} is connected to the cell uid={1} but the cell"
                       " is not connected to this edge.",ItemPrinter(edge),ItemPrinter(cell));
        }
      }
    }
  }
  // Vérifie que la connective maille<->face
  // est réciproque
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
	  Face face = *iface;
	  if(!m_mesh->isAmrActivated()){
		  for( Cell cell : face.cells() ){
			  bool has_found = false;
			  for( Face face2 : cell.faces() ){
				  if (face2==face){
					  has_found = true;
					  break;
				  }
			  }
			  if (!has_found){
          ARCANE_FATAL("Face uid={0} is connected to the cell uid={1} but the cell"
                       " is not connected to this face.",ItemPrinter(face),ItemPrinter(cell));
			  }
		  }
	  }
    for( Cell cell : face.cells() ){
      bool has_found = false;
      for( Face face2 : cell.faces() ){
        if (face2==face){
          has_found = true;
          break;
        }
      }
      if (!has_found && (face.backCell().level() == face.frontCell().level())){
        warning() << func_name << ". The face " << FullItemPrinter(face)
                  << " is connected to the cell " << FullItemPrinter(cell)
                  << " but the cell (level "<< cell.level()<< ")"
                  << " is not connected to the face.";
      }
    }

	  Integer nb_cell = face.nbCell();
	  Cell back_cell = face.backCell();
	  Cell front_cell = face.frontCell();
	  if ((back_cell.null() || front_cell.null()) && nb_cell==2)
		  ARCANE_FATAL("Face uid='{0}' bad number of cells face",ItemPrinter(face));
	  // Si on a deux mailles connectées, alors la back cell doit être la première
	  if (nb_cell==2 && back_cell!=face.cell(0))
		  ARCANE_FATAL("Bad face face.backCell()!=face.cell(0) face={0} back_cell={1} from_cell={2} cell0={3}",
                   ItemPrinter(face),ItemPrinter(back_cell),ItemPrinter(front_cell),ItemPrinter(face.cell(0)));
  }

  _checkFacesOrientation();

  if (m_mesh->parentMesh()){
    Integer nerror = 0;
    eItemKind kinds[] = { IK_Node, IK_Edge, IK_Face, IK_Cell };
    Integer nb_kind = sizeof(kinds)/sizeof(eItemKind);

    for(Integer i_kind=0;i_kind<nb_kind;++i_kind){
      const eItemKind kind = kinds[i_kind];
      IItemFamily * family = m_mesh->itemFamily(kind);
      if (!family->parentFamily()){
        error() << "Mesh " << m_mesh->name() << " : Family " << kind << " does not exist in mesh";
        ++nerror;
      }
      else{
        ENUMERATE_ITEM(iitem,family->allItems()){
          const Item & item = *iitem;
          const Item & parent_item = item.parent();
          if (parent_item.itemBase().isSuppressed()){
            error() << "Mesh " << m_mesh->name() << " : Inconsistent suppresssed parent item uid : " 
                    << ItemPrinter(item) << " / " << ItemPrinter(parent_item);
            ++nerror;
          }
          if (parent_item.uniqueId() != item.uniqueId()){
            error() << "Mesh " << m_mesh->name() << " : Inconsistent item/parent item uid : " << ItemPrinter(item);
            ++nerror;
          }
        }
      }
    }
    if (nerror > 0)
      ARCANE_FATAL("Mesh name={0} has {1} (see above)",m_mesh->name(),String::plural(nerror, "error"));

    // Vérifie la consistence parallèle du groupe parent
    nerror = 0;
    {
      ItemGroup parent_group = m_mesh->parentGroup();
      String var_name = String::format("SubMesh_{0}_GroupConsistencyChecker",parent_group.name());
      TemporaryVariableBuildInfo tvbi(m_mesh->parentMesh(),var_name,parent_group.itemFamily()->name());
      ItemVariableScalarRefT<Integer> var(tvbi,m_mesh->parentGroup().itemKind());
      var.fill(-1);
      ENUMERATE_ITEM(iitem, m_mesh->parentGroup()){
        var[iitem] = iitem->owner();
      }
      nerror += var.checkIfSync(10);
    }
    if (nerror > 0)
      ARCANE_FATAL("Mesh name={0} has parent group consistency {1}\n"
                   "This usually means that parent group was not symmetrically built",
                   m_mesh->name(),String::plural(nerror, "error", false));
  }
  if (m_is_check_items_owner){
    _checkValidItemOwner(m_mesh->nodeFamily());
    _checkValidItemOwner(m_mesh->edgeFamily());
    _checkValidItemOwner(m_mesh->faceFamily());
    _checkValidItemOwner(m_mesh->cellFamily());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
updateAMRFaceOrientation()
{
	String func_name = "MeshChecker::updateAMRFaceOrientation";
  FaceReorienter fr(m_mesh);
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    Integer nb_cell = face.nbCell();
    Cell back_cell = face.backCell();
    Cell front_cell = face.frontCell();
    if ((back_cell.null() || front_cell.null()) && nb_cell==2)
      ARCANE_FATAL("Bad number of cells for face={0}",ItemPrinter(face));
    // Si on a deux mailles connectées, alors la back cell doit être la première
    if (nb_cell==2 && back_cell!=face.cell(0))
		  ARCANE_FATAL("Bad face face.backCell()!=face.cell(0) face={0} back_cell={1} from_cell={2} cell0={3}",
                   ItemPrinter(face),ItemPrinter(back_cell),ItemPrinter(front_cell),ItemPrinter(face.cell(0)));

    // Ceci pourrait sans doutes etre optimise
    fr.checkAndChangeOrientationAMR(face);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
updateAMRFaceOrientation(ArrayView<Int64> ghost_cell_to_refine)
{
  ItemInternalList cells = m_mesh->itemsInternal(IK_Cell) ;
  UniqueArray<Integer> lids(ghost_cell_to_refine.size()) ;
  m_mesh->cellFamily()->itemsUniqueIdToLocalId(lids,ghost_cell_to_refine,true) ;
  FaceReorienter fr(m_mesh);
  std::set<Int64> set ;
  typedef std::pair<std::set<Int64>::iterator,bool> return_type ;
  for(Integer i=0, n=lids.size();i<n;++i){
    Cell icell = cells[lids[i]] ;
    for( Integer ic=0, nchild=icell.nbHChildren();ic<nchild;++ic){
      Cell child = icell.hChild(ic) ;
      for( Face face : child.faces() ){
        return_type value = set.insert(face.uniqueId());
        if(value.second){
          fr.checkAndChangeOrientationAMR(face);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les faces sont correctement orientées et connectées.
 */
void DynamicMeshChecker::
_checkFacesOrientation()
{
  bool is_1d = (m_mesh->dimension()==1);
  //Temporaire: pour l'instant, ne teste pas l'orientation en 1D.
  if (is_1d)
    return;

  String func_name = "MeshChecker::_checkFacesOrientation";

  Int64UniqueArray work_face_sorted_nodes;
  IntegerUniqueArray work_face_nodes_index;
  Int64UniqueArray work_face_orig_nodes_uid;
  ItemTypeMng* item_type_mng = m_mesh->itemTypeMng();
  NodesOfItemReorderer face_reorderer(item_type_mng);
  ENUMERATE_(Cell,icell,m_mesh->allCells()){
    Cell cell = *icell;
    const ItemTypeInfo* type = cell.typeInfo();
    Int32 cell_nb_face = type->nbLocalFace();
    for( Integer i_face=0; i_face<cell_nb_face; ++i_face ){
      const ItemTypeInfo::LocalFace& lf = type->localFace(i_face);
      Integer face_nb_node = lf.nbNode();

      work_face_orig_nodes_uid.resize(face_nb_node);
      for( Integer z=0; z<face_nb_node; ++z )
        work_face_orig_nodes_uid[z] = cell.node(lf.node(z)).uniqueId();

      bool is_reorder = false;
      if (is_1d){
        is_reorder = face_reorderer.reorder1D(i_face, work_face_orig_nodes_uid[0]);
      }
      else
        is_reorder = face_reorderer.reorder(ItemTypeId::fromInteger(lf.typeId()), work_face_orig_nodes_uid);
      ConstArrayView<Int64> face_sorted_nodes(face_reorderer.sortedNodes());

      Face cell_face = cell.face(i_face);
      if (cell_face.nbNode()!=face_nb_node)
        ARCANE_FATAL("Incoherent number of node for 'face' and 'localFace'"
                     " cell={0} face={1} nb_local_node={2} nb_face_node={3}",
                     ItemPrinter(cell),ItemPrinter(cell_face),face_nb_node,cell_face.nbNode());

      for( Integer z=0; z<face_nb_node; ++z ){
        if (cell_face.node(z).uniqueId()!=face_sorted_nodes[z])
          ARCANE_FATAL("Bad node unique id for face: cell={0} face={1} cell_node_uid={2} face_node_uid={3} z={4}",
                       ItemPrinter(cell),ItemPrinter(cell_face),face_sorted_nodes[z],
                       cell_face.node(z).uniqueId());
      }
      if (is_reorder){
        Cell front_cell = cell_face.frontCell();
        if (front_cell!=cell){
          if (!front_cell.null())
            ARCANE_FATAL("Bad orientation for face. Should be front cell: cell={0} face={1} front_cell={2}",
                         ItemPrinter(cell),ItemPrinter(cell_face),ItemPrinter(front_cell));
          else
            ARCANE_FATAL("Bad orientation for face. Should be front cell (no front cell) cell={0} face={1}",
                         ItemPrinter(cell),ItemPrinter(cell_face));
        }
      }
      else{
        Cell back_cell = cell_face.backCell();
        if (back_cell!=cell){
          if (!back_cell.null())
            ARCANE_FATAL("Bad orientation for face. Should be back cell: cell={0} face={1} front_cell={2}",
                         ItemPrinter(cell),ItemPrinter(cell_face),ItemPrinter(back_cell));
          else
            ARCANE_FATAL("Bad orientation for face. Should be back cell (no back cell) cell={0} face={1}",
                         ItemPrinter(cell),ItemPrinter(cell_face));
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
_checkValidItemOwner(IItemFamily* family)
{
  // Pour les maillages non sous-maillages, il faut que tout sub-item
  // (Node, Edge ou Face) ait une cellule voisine de même propriétaire,
  // sauf si on autorise les entités orphelines et que l'entité n'est
  // connectée à aucune maille.
  // Pour les sous-maillages, il faut, en plus, que tout item soit de
  // même propriétaire que son parent.
  bool allow_orphan_items = m_mesh->meshKind().isNonManifold();

  Integer nerror = 0;
  if (!m_mesh->parentMesh()){
    
    if (family->itemKind() == IK_Cell)
      return; // implicitement valide pour les cellules
    
    ItemGroup own_items = family->allItems().own();
    ENUMERATE_ITEM(iitem,own_items){
      Item item = *iitem;
      Int32 owner = item.owner();
      bool is_ok = false;
      ItemVectorView cells = item.itemBase().cellList();
      if (cells.size()==0 && allow_orphan_items)
        continue;
      for( Item cell : cells ){
        if (cell.owner()==owner){
          is_ok = true;
          break;
        }
      }
      if (!is_ok) {
        OStringStream ostr;
        Integer index = 0;
        ostr() << " nb_cell=" << cells.size();
        for( Item cell : cells ){
          ostr() << " SubCell i=" << index << " cell=" << ItemPrinter(cell);
          ++index;
        }
        error() << "Mesh " << m_mesh->name() << " family=" << family->name()
                << " Item" << ItemPrinter(item) << " has no cell with same owner:"
                << ostr.str();
        ++nerror;
      }
    }
  }
  else {
    ENUMERATE_ITEM(iitem,family->allItems()){
      Item item = *iitem;
      Item parent_item = item.parent();
      if (parent_item.owner() != item.owner()) {
        error() << "Mesh " << m_mesh->name() << " : Inconsistent item/parent item owner : "
                << ItemPrinter(item) << " / " << ItemPrinter(parent_item);
        ++nerror;
      }
    }
  }
  
  if (nerror>0)
    ARCANE_FATAL("mesh {0} family={1} has {2}",m_mesh->name(),family->name(),
                 String::plural(nerror,"owner error"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
checkVariablesSynchronization()
{
  Int64 nb_diff = 0;
  VariableCollection used_vars(m_mesh->variableMng()->usedVariables());
  for( VariableCollection::Enumerator i_var(used_vars); ++i_var; ){
    IVariable* var = *i_var;
    switch (var->itemKind()){
    case IK_Node:
    case IK_Edge:
    case IK_Face:
    case IK_Cell:
    case IK_DoF:
      nb_diff += var->checkIfSync(10);
      break;
    case IK_Particle:
    case IK_Unknown:
      break;
    }
  }
  if (nb_diff!=0)
    ARCANE_FATAL("Error in checkVariablesSynchronization() nb_diff=",nb_diff);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
checkItemGroupsSynchronization()
{
  Int64 nb_diff = 0;
  // TODO: parcourir toutes les familles (sauf particules)
  ItemGroupsSynchronize node_sync(m_mesh->nodeFamily());
  nb_diff += node_sync.checkSynchronize();
  ItemGroupsSynchronize edge_sync(m_mesh->edgeFamily());
  nb_diff += edge_sync.checkSynchronize();
  ItemGroupsSynchronize face_sync(m_mesh->faceFamily());
  nb_diff += face_sync.checkSynchronize();
  ItemGroupsSynchronize cell_sync(m_mesh->cellFamily());
  nb_diff += cell_sync.checkSynchronize();
  if (nb_diff!=0)
    ARCANE_FATAL("some groups are not synchronized nb_diff={0}",nb_diff);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que la couche de mailles fantômes est correcte.
 *
 * Vérifie que toutes les mailles fantômes sont bien connectés à
 * une maille de ce sous-domaine (cas où il n'y a qu'une couche de mailles
 * fantômes).
 * \todo Vérifier qu'aucune maille du bord n'appartient à ce sous-domaine
 * \todo Supporter plusieurs couches de mailles fantômes.
 */
void DynamicMeshChecker::
checkGhostCells()
{
  pwarning() << "CHECK GHOST CELLS";
  Integer sid = m_mesh->meshPartInfo().partRank();
  ENUMERATE_CELL (icell, m_mesh->cellFamily()->allItems()) {
    Cell cell = *icell;
    if (cell.isOwn())
      continue;
    bool is_ok = false;
    for (Node node : cell.nodes()) {
      for (Cell cell2 : node.cells()) {
        if (cell2.owner() == sid) {
          is_ok = true;
        }
      }
    }
    if (!is_ok)
      info() << "WARNING: Cell " << ItemPrinter(cell) << " should not be a ghost cell";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
checkMeshFromReferenceFile()
{
  if (!m_compare_reference_file)
    return;

  IParallelMng* pm = m_mesh->parallelMng();

  if (!pm->isParallel())
    return; // uniquement en parallèle

  debug() << "Testing the mesh against the initial mesh";
  String base_file_name("meshconnectivity");
  // En parallèle, compare le maillage actuel avec le fichier
  // contenant la connectivité complète (cas séquentiel)
  IIOMng* io_mng = pm->ioMng();
  ScopedPtrT<IXmlDocumentHolder> xml_doc(io_mng->parseXmlFile(base_file_name));
  if (xml_doc.get()){
    XmlNode doc_node = xml_doc->documentNode();
    if (!doc_node.null())
      mesh_utils::checkMeshConnectivity(m_mesh,doc_node,true);
  }
  else{
    warning() << "Can't test the subdomain coherence "
              << "against the initial mesh";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
checkValidReplication()
{
  info() << "Checking valid replication";
  IParallelMng* mesh_pm = m_mesh->parallelMng();
  IParallelReplication* pr = mesh_pm->replication();
  if (!pr->hasReplication())
    return;

  IParallelMng* pm = pr->replicaParallelMng();

  // Vérifie que toutes les familles (sauf les particules) sont les mêmes
  UniqueArray<IItemFamily*> wanted_same_family;

  for( IItemFamilyCollection::Enumerator i(m_mesh->itemFamilies()); ++i; ){
    IItemFamily* family = *i;
    if (family->itemKind()!=IK_Particle)
      wanted_same_family.add(family);
  }
  ValueChecker vc(A_FUNCINFO);

  // Vérifie que tout le monde à le même nombre de famille.
  Integer nb_family = wanted_same_family.size();
  Integer max_nb_family = pm->reduce(Parallel::ReduceMax,nb_family);
  vc.areEqual(nb_family,max_nb_family,"Bad number of family");

  // Vérifie que toutes les familles ont le même nombre d'éléments.
  //TODO: il faudrait vérifier aussi que les noms des familles correspondent.
  {
    UniqueArray<Int32> families_size(nb_family);
    for( Integer i=0; i<nb_family; ++i )
      families_size[i] = wanted_same_family[i]->nbItem();
    UniqueArray<Int32> global_families_size(families_size);
    pm->reduce(Parallel::ReduceMax,global_families_size.view());
    vc.areEqualArray(global_families_size,families_size,"Bad family");
  }

  // Vérifie que toutes les familles ont les mêmes entités (même uniqueId())
  {
    UniqueArray<Int64> unique_ids;
    UniqueArray<Int64> global_unique_ids;
    for( Integer i=0; i<nb_family; ++i ){
      ItemGroup group = wanted_same_family[i]->allItems();
      unique_ids.resize(group.size());
      Integer index = 0;
      ENUMERATE_ITEM(iitem,group){
        unique_ids[index] = iitem->uniqueId();
        ++index;
      }
      global_unique_ids.resize(group.size());
      global_unique_ids.copy(unique_ids);
      pm->reduce(Parallel::ReduceMax,global_unique_ids.view());
      String message = String::format("Bad unique ids for family '{0}'",
                                      wanted_same_family[i]->name());
      vc.areEqualArray(global_unique_ids,unique_ids,message);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshChecker::
_checkReplicationFamily(IItemFamily*)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
